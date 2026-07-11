# G05 Isaac Training First Loop

Status: Partial

## Goal

Run the first end-to-end Isaac training loop that produces a policy artifact, even if the policy is weak.

This gate should prove that the training stack can consume A3 assets and produce a deployable policy format.

## Inputs

- Isaac-ready A3 asset from G04.
- Motion references or placeholder task references.
- BeyondMimic/whole-body tracking scaffold.
- Policy observation/action contract.

## Outputs

- First accepted training run.
- Training config.
- Logs and metrics.
- Exported policy artifact path or metadata.
- Initial policy evaluation notes.

## Related Directories

- `hope_training/whole_body_tracking`
- `docs/interfaces/policy_observation_action.md`
- `docs/operations/run_training.md`
- `vendor_assets/` for generated heavy policy artifacts if needed
- `external_repos/TTRL-ICRA2026` as an auto-synced reference if first-loop failures need comparison

## Operation Docs

- [../operations/run_training.md](../operations/run_training.md)
- [../operations/setup_local_sync.md](../operations/setup_local_sync.md)

## Acceptance Criteria

- Isaac environment starts with the A3 asset.
- A random rollout works.
- A first PPO or equivalent training loop runs.
- Policy export path is documented.
- First-loop results are recorded, even if poor.

## Current State

Follow-up note (2026-07-05, R15 v5 correction):

- Reverted the mistaken v5 default switch in `/workspace/yikang/nohope`: `cfg/train.yaml` and `cfg/play.yaml` keep `motion_file: null`, and the product/default deploy-parity task remains on the hopex lineage (`strike_phase_per_clip: [0.47, 0.333]` with the existing hopex target boxes). v5 is an R15 ablation arm only, passed with explicit `motion_file=` / `motion_file_2=` and `task.racket.*` CLI overrides.
- Added `hope_training/whole_body_tracking/cfg/strike_annotations.yaml` as the source of truth for hand-aligned contact phases. `scripts/analyze_strike_phase.py` now treats speed-peak picks as diagnostic candidates and applies annotations first. The checked v5 output selects forehand frame 37 / phase 0.673, not the frame 43/44 post-contact whip (`vx` near zero, +Y-heavy); backhand phase 0.345 remains unverified until a frame-by-frame scrub.
- R15 v5 sampling boxes are generated from the annotated frames: forehand pos x/y/z `[0.29,0.49]`, `[-0.63,-0.43]`, `[0.74,0.94]`; forehand vel x/y/z `[0.74,1.74]`, `[0.71,1.71]`, `[1.20,2.20]`; backhand pos x/y/z `[0.60,0.80]`, `[0.12,0.32]`, `[0.81,1.01]`; backhand vel x/y/z `[2.60,3.60]`, `[0.50,1.50]`, `[1.66,2.66]`. Face normals from video/GVHMR clips are wrist +Y proxies and marked unreliable.

Follow-up note (2026-07-02/03, after the simtoreal2 merge):

- The training default is now `task=HOPEPingPongDeployParity` (`cfg/train.yaml`;
  `HOPEPingPongRealSensor` is a backward-compat alias): the 175-D deploy-parity actor contract
  (see [../interfaces/policy_observation_action.md](../interfaces/policy_observation_action.md)),
  base-free footwork rewards (`base_position` removed; dense `racket_progress` + pre-strike
  slip/twist/upright penalties + `arm_torque_saturation`), per-clip blade-centered 3-D target
  pos/vel boxes, `strike_phase_per_clip: [0.47, 0.333]` on the re-grounded `_hopex` (v3) clips,
  `racket_velocity_std: 1.0` (plan 1.0 → 0.8 → 0.5), and PD-gain DR re-enabled at ±15%
  (2026-07-02 sim2real fine-tune; documented HITTER departure). The 180-D `task=HOPEPingPong` is a
  legacy comparison path and is not deploy-honest.
- This path produced the first hardware-deployed policy: `model_p4_deployparity.onnx` (175-D /
  31-act), sim2sim-validated in MuJoCo and run on the real A3 on 2026-07-02 (forehand only). The
  newest lineage is the explicit-clipped-PD fine-tune (`launch_explicitpd_ft.sh`, model_25700).
  Contract checks: `hope_isaac_py scripts/verify_realsensor.py --check layout|rollout|onnx`.
- Exact accepted run IDs/metrics for a quality baseline are still pending (see Not done).

Follow-up note (2026-07-01, `main` after the unified HITTER audit — values superseded above):

- The active `HOPEPingPong` config then defaulted to unified forehand+backhand training (`registry_name_2` enabled), `target_mode: uniform`, fixed strike plane `x=0.4`, `strike_phase_per_clip: [0.36, 0.74]` (v1 clips), actor `swing_type`, and no actor `racket_target_normal_w`.
- The `registry_for_runner` blocker and local-motion regression found during the audit were fixed in the training entry: local `motion_file=<forehand.npz> motion_file_2=<backhand.npz>` now bypasses WandB, while registry-backed runs still link the used registry artifact(s).
- The 2026-06-26 first-loop result remains useful as pipeline history, but the unified HOPEPingPong path still needs a fresh Isaac run before it can count as an accepted baseline.

Done:

- Training scaffold is present under `hope_training/whole_body_tracking`.
- Existing docs describe BeyondMimic-style training assumptions.
- The branch adds Hydra training/eval entrypoints, HOPEPingPong task config, racket-target command logic, and A3-specific robot config.
- `reimplement.md` records that `TrackingFlat` and `HOPEPingPong` forehand training have run end-to-end on the copied A3 URDF asset, including wandb logging, checkpoint save, and `policy.onnx` export.
- Commit `42489cd` adds `setup_train_env.sh`, richer WandB/live metrics, and in-container ONNX export support.
- Earlier `reference_perturbed` / PATH B-C experiments (commit `c951d9d`) are present in history, but the current default is the 2026-07-01 unified HITTER path: direct uniform target sampling, no success-gated perturbation curriculum, and no `ref_vel_scale` ramp. `reference_perturbed` remains available as a non-default `target_mode` and now uses per-clip reference strike centers.
- `RacketTargetCommand` logs conditional exact-strike pass rates: `strike_pos_pass_exact`, `strike_vel_pass_exact`, `strike_normal_pass_exact`, `strike_composite_success_exact`, and `exact_strike_sample_count_decayed`.
- `RacketTargetCommand` also supports optional debug reward logging (`debug_reward_logging`) for swing-through sign checks and raw-vs-gated reward kernels. Keep it off for production runs unless diagnosing reward scale.
- The HOPE actor observation includes `swing_type` and desired runtime targets, but not `racket_target_normal_w`; racket pose/velocity/normal remain critic/reward-only simulation state.
- `scripts/train.py` logs import provenance, env-cfg source, every applied task override, and post-override racket knobs; YAML keys that target missing env-cfg attributes raise instead of silently no-oping.
- `scripts/train.py` keeps registry defaults available from `cfg/task/*.yaml`, while `motion_file=<local.npz>` and optional `motion_file_2=<local.npz>` take precedence for no-WandB smoke tests or locally generated references.
- Local unified-policy training can use Step 9-12 video-generated motions directly with `motion_file=../motions/preprocessed/hope_forehand.npz motion_file_2=../motions/preprocessed/hope_backhand.npz logger=tensorboard`.
- Generated ONNX policy artifacts remain ignored by asset policy unless a gate records an external artifact path.
- Merged from `train_1` (2026-06-26) and superseded by the unified HITTER alignment: paddle-contact timing is per clip, expressed as `strike_phase_per_clip` (then `[0.36, 0.74]` on v1 clips; current default `[0.47, 0.333]` on the `_hopex` v3 clips); `episode_length_s: 3.0` caps each episode to about one swing; `scripts/train.py` / `cfg/train.yaml` keep the `checkpoint_path` knob for staged resume.
- 2026-07-02: `HOPEPingPong.yaml` and `HOPEPingPongRealSensor.yaml` were synchronized for unified forehand+backhand training. Old single-swing wording and old backhand positive-y/high-z coordinate notes were removed from the YAML comments; both observation-route candidates now point at the same re-grounded target boxes.
- 2026-07-02: Clip wrap no longer performs mid-episode RSI for HOPE ping-pong (`motion.rsi_on_wrap: false` in both task YAMLs; the knob was renamed to main's equivalent `motion.wrap_teleport` on 2026-07-03). Episode reset still initializes from the reference, but wrap only advances the reference clip/time and target, forcing the policy to learn physical between-swing recovery.
- 2026-07-02: `racket_progress` now resets its previous-distance baseline and emits zero on motion/target resample steps, removing the fixed wrap/reset reward spike from the base-free footwork signal.
- 2026-07-02: `HOPEPingPong` and `HOPEPingPongRealSensor` were switched to `racket.target_mode: reference_perturbed`: the initial target center is the imitated clip's own strike-frame racket FK state, and the long-run distribution widens through success-gated perturbations (`ref_perturb_pos=[0.15,0.20,0.15]`, `ref_perturb_vel=[1.0,1.0,0.8]`). [Pre-merge branch configuration: after merging main's unified HITTER redesign, the default is `target_mode: uniform` with a fixed strike plane; `reference_perturbed` and these perturbation ranges remain available as a non-default option.]
- 2026-07-02: The shared RunPod now has an independent `hope-motion-py310` Conda env for GVHMR/GMR motion preparation, separate from Isaac Lab. GVHMR/GMR import checks pass on RTX 5090 with PyTorch `2.7.0+cu128`; PyTorch3D `0.7.9` was built from source for `sm_120`; GVHMR non-body checkpoints are present; and the local ignored GMR clone has verified `agibot_a3` MJCF/IK registration with 31 hinge joints matching `joint_order_agibot_a3.yaml`.
- 2026-07-02: Uploaded forehand/backhand MP4s were converted through GVHMR -> GMR -> `scripts/csv_to_npz.py` into local ignored motions: `hope_training/motions/preprocessed/hope_forehand.npz` (`joint_pos=(139,31)`, `body_pos_w=(139,32,3)`, `fps=50`) and `hope_training/motions/preprocessed/hope_backhand.npz` (`joint_pos=(132,31)`, `body_pos_w=(132,32,3)`, `fps=50`).
- 2026-07-02: WandB setup is verified for `WANDB_ENTITY=BerkeleyPingPong`, `WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org`, `WANDB_PROJECT=hope_wbc`, and `WANDB_MOTION_PROJECT=csv_to_npz`. The registry aliases `dongc_1-university-of-california-berkeley-org/wandb-registry-motions/hope_forehand:latest` and `.../hope_backhand:latest` resolve to `BerkeleyPingPong/csv_to_npz/hope_forehand:v4` and `BerkeleyPingPong/csv_to_npz/hope_backhand:v4`; both contain `motion.npz`.
- 2026-07-02: Registry-backed `HOPEPingPong` WandB smoke passed with `source setup_train_env.sh && hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true num_envs=32 max_iterations=1 logger=wandb run_name=smoke_registry_wandb_finish`. W&B run `6xus13ga` finished at https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/6xus13ga, used both motion artifacts, and synced `model_0.pt`, `2026-07-02_11-56-04_smoke_registry_wandb_finish.onnx`, config, diff, output log, and summary. Local outputs are under `hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope/2026-07-02_11-56-04_smoke_registry_wandb_finish/`.
- 2026-07-02: The same registry `hope_forehand/backhand:v4` artifacts used for the smoke were later verified as directionally wrong for real training: frame-0 pelvis yaw is 82.03/85.92 deg and strike velocity is +Y-dominant. Corrected local ignored clips were generated at `hope_training/motions/preprocessed/hope_forehand_hopex.npz` and `hope_backhand_hopex.npz`. [Pre-merge, both task YAMLs defaulted to these local files; post-merge the YAMLs default to the WandB registry aliases again — pass `motion_file=`/`motion_file_2=` to train on the corrected `_hopex` clips until a future verified registry artifact replaces them. The local v5 clips are R15-only, not that replacement.]
- 2026-07-02: `scripts/csv_to_npz.py --robot agibot_a3` now auto-aligns exported world-frame arrays into HOPE +X before saving/uploading, and `scripts/check_motion_target_alignment.py` provides a no-Isaac gate. Verification: `python -m py_compile ...` passed; `python scripts/check_motion_target_alignment.py --yaml cfg/task/HOPEPingPong.yaml` and `--yaml cfg/task/HOPEPingPongRealSensor.yaml` passed; the same check fails on old v4 as expected.
- 2026-07-02 (later): Fixed an `UnboundLocalError` in `RacketTargetCommand._resample_command` (`hope_commands.py`; the base-XY coupling branch read `motion` before assignment) that crashed EVERY `HOPEPingPong*` env reset — the working tree could not start training at all until this fix. After the fix: local-clip smoke passed (`num_envs=32 max_iterations=2 logger=tensorboard`), and a bounded verification training run passed: `hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true num_envs=4096 max_iterations=300 run_name=e2e_verify_train seed=1` -> W&B run `wuj6ds9u` (https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/wuj6ds9u), mean reward -1.37 -> ~25, mean episode length 5 -> ~340 steps, `strike_success` 0 -> 0.006 in 300 iters, `model_{0,100,200,299}.pt` + ONNX exported and synced. Pipeline viability evidence on the corrected `_hopex` clips, still not a quality baseline. [This run used the pre-merge branch defaults, i.e. `target_mode: reference_perturbed`.]
- 2026-07-02 (later): Full fresh MP4 -> npz rerun in an isolated dir reproduces the shipped artifacts bit-for-bit (GMR pkl and retargeted CSV byte-identical; npz equal to `hope_forehand_hopex.npz` within 2e-7 float noise), and `check_motion_target_alignment.py --clip` passes on the regenerated clip. One env caveat found and documented: GVHMR's YOLO load needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` under torch 2.7 (see `docs/operations/setup_environments.md`).
- 2026-07-04: `motion.clip_switch_prob` (default 0.0, try 0.002) adds deploy-parity MID-swing clip
  switches through the existing wrap-resample path: per-step random abort to a different clip's
  frame 0 + fresh pre-swing hold + fresh target, robot untouched. Parity for
  `pp_reference_clock.hpp`, which flips `clip_id` at arbitrary tts when the planner re-sides the
  target — the root cause of the venue falls at 准备/正手/反手 switches. A8 post-swing capture
  stays wrap-only (aborted swings are not captured). Mech-verified via `clip_switch_count`.
- 2026-07-04: P2.4 `base_decel` reward landed default-off (`rewards.base_decel_weight: 0.0`;
  PACE-style pre-strike pseudo-speed tracking on racket→target planar distance — formula and v2
  plan in `docs/operations/run_training.md` Reward Shaping and `docs/motion_and_contract_v3.md`
  §5). Mech-verified on/off. Same commit hardened `scripts/train.py`: `task.motion`/`task.racket`
  yaml keys go through explicit whitelists and unknown keys RAISE — new yaml keys must extend the
  whitelist in the same commit (the 018467a startup-crash lesson).
- 2026-07-04 (evening): `motion.speed_scale_range` (R14 retiming) landed default-off — per-swing
  reference playback speed with the full consistency cascade (clock ×s via float shadow clock,
  reference velocities ×s, tts ÷s, racket velocity target ×s incl. HER clamp box); train-only
  (play/eval force `[1.0, 1.0]`); flag docs in `docs/operations/run_training.md`, design in
  `docs/motion_and_contract_v3.md` §2. MECH-VERIFIED on pod (2026-07-04 late): OFF run 25 it
  clean; ON run `[0.8,1.2]` 25 it clean with `Live/motion/playback_speed` per-iter env-mean
  fluctuating 0.981-1.012 — the U(0.8,1.2)/√512 signature (a dead flag would sit at 1.0000).
  Arm R14 is launch-ready.
- 2026-07-04 (late): `rewards.free_wrist_ori_mimic` (R16, franco's wrist idea) landed default-off
  and MECH-VERIFIED on pod (25 it clean; startup override log shows both
  `motion_body_ori.body_names-=right_wrist_yaw_Link` and `motion_body_ang_vel.body_names-=...`).
  Config-level: this codebase has no joint-level mimic rewards — body-level orientation tracking
  on the racket-mount link IS the face mimic; the flag filters it out of the two orientation
  terms while keeping position/linear-velocity mimic (swing path). Face is then shaped by
  `racket_normal` / ball-outcome rewards; at contract v3 the freed wrist becomes the actuator of
  the commanded-normal channel.
- 2026-07-04 (evening): v5 clips processed end-to-end ON the pod (`v5_pipeline.sh`, reusing the
  oblique pipeline + `csv_to_npz_mujoco.py`): `/workspace/shared/motions/hope_{forehand,backhand}_v5.npz`
  (56/58 frames @50 Hz, yaw re-grounded +86.6°/+83.6°→0). Strike phases CORRECTED late-night by
  franco's prior: forehand 0.673 (detector's speed-peak 0.768 is the post-contact whip — at the
  true ~2/3 contact the velocity/normal are direction-healthy, retiring the "+Y-dominant" flag),
  backhand 0.345 (matches franco's "within the first 3/7"). Lesson: speed peak != contact;
  cross-check `analyze_strike_phase` picks against the forward-velocity peak and a human prior.
  Remaining data flag: v5 reference jitter is 2-6× hopex (mean joint |acc| 5.9/15.5 vs 2.5/2.7
  rad/s²; oblique 3.5 sits between) — evidence for R16 / reference filtering, and a third
  confounder for R15 verdicts.

Done (2026-06-26 — first loop reproduced in this harness):

- The Isaac WBC training loop now runs end-to-end on this machine. Run: task `TrackingFlat`, `num_envs=1024`, `max_iterations=60`, `algo.runner.save_interval=25`, `logger=tensorboard`, `run_name=stand_bootstrap`. Mean reward improved monotonically `-4.08 -> -0.24`. Artifacts under `hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_flat/2026-06-26_13-13-07_stand_bootstrap/`: checkpoints `model_0.pt`, `model_25.pt`, `model_50.pt`, `model_59.pt`, and `exported/policy.onnx` (exported via `scripts/play.py task=TrackingFlat ... checkpoint=model_59.pt motion_file=../motions/a3_stand.npz`). These prove pipeline viability only; the reference clip is a static stand, not a swing.
- Two blockers were resolved to get here, beyond EULA acceptance:
  1. **Blackwell GPU incompatibility (the real blocker).** The RTX 5090 is sm_120; Isaac Sim 4.5.0's bundled `torch 2.5.1+cu124` has no sm_120 kernels and a real CUDA matmul fails with `no kernel image is available for execution on the device`. Fixed by upgrading `hope-isaac-py310` to `torch 2.7.0+cu128` / `torchvision 0.22.0+cu128` and pinning `numpy==1.26.4` (Isaac needs `numpy<2`). Rollback: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124`. `isaaclab*` keep a `torch==2.5.1` pin in metadata but are editable installs imported at runtime, so the upgrade does not break them.
  2. **WandB-only motion loading.** `scripts/train.py` fetched the motion clip only from the WandB registry. Added a local `motion_file=` override (skips WandB, mirrors `play.py`) and `motion_file: null` in `cfg/train.yaml`.
- EULA accepted non-interactively via `OMNI_KIT_ACCEPT_EULA=YES`.
- Bootstrap motion: `scripts/make_static_motion.py` generates `hope_training/motions/a3_stand.npz` (static default-pose clip, `fps=50`, `joint_pos[600,31]`, `body_pos_w[600,32,3]`) so the loop runs without the GMR/GVHMR pipeline or WandB. Placeholder reference only.

Not done:

- No accepted quality baseline is set yet; the recorded local and WandB smoke runs and the bounded `wuj6ds9u` verification run prove pipeline viability, not policy strength.
- The verified WandB smoke used only 32 envs for 1 PPO iteration and used later-rejected +Y-facing registry motions; no long unified policy training run or post-train evaluation has been accepted.
- The post-merge unified uniform-target configuration has not been run in Isaac yet; the 2026-07-02 verification runs used this branch's pre-merge `reference_perturbed` defaults, so the unified HOPEPingPong path still needs a fresh run before it can count toward a baseline.
- Uniform reachable target ranges, reward tuning, exact-strike pass rates, stable recovery metrics, and first usable baseline thresholds still need formal acceptance.
- Exact accepted run IDs, checkpoint paths, ONNX paths, and first quality metrics still need to be recorded in this gate for an accepted run.
- The corrected `_hopex.npz` motion clips are ignored local artifacts; new machines must restore or regenerate them through `setup_local_sync.md` before reproducing the recorded local-clip `HOPEPingPong*` training runs.
- The v5 R15 ablation clips have no accepted smoke, training run, or quality baseline; forehand is hand-verified at phase 0.673, while backhand phase 0.345 is still unverified.

## Current Verification Commands

Account-free reproduction (no WandB, no motion data — the 2026-06-26 path). On a Blackwell GPU apply the
torch-cu128 fix first (see [run_training.md](../operations/run_training.md#blackwell-rtx-50-series-sm_120-torch-fix)):

```bash
export OMNI_KIT_ACCEPT_EULA=YES
hope_isaac_py scripts/make_static_motion.py --robot agibot_a3 \
  --output_file ../motions/a3_stand.npz --frames 600 --fps 50
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  num_envs=1024 max_iterations=60 algo.runner.save_interval=25 \
  logger=tensorboard run_name=stand_bootstrap \
  motion_file=$(pwd)/../motions/a3_stand.npz
```

GPU/Isaac environment, after `source setup_train_env.sh` and restoring/generating Step 9-12 local motions:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke

hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  motion_file_2=../motions/preprocessed/hope_backhand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=hope_smoke
```

Note: if the local Step 9-12 clips predate the 2026-07-02 HOPE +X alignment fix in `csv_to_npz.py`, use the
corrected `hope_forehand_hopex.npz` / `hope_backhand_hopex.npz` clips (or regenerate);
`scripts/check_motion_target_alignment.py --clip <npz>` verifies alignment without Isaac.

Record the startup lines from `scripts/train.py` showing source provenance and applied overrides. For
an accepted run, also record the registry artifact or local `motion_file`, WandB run ID when logging to
WandB, checkpoint path, exported ONNX path, and exact-strike pass metrics.

### Local Harness Check 2026-06-25

Commands run from the repo root:

```bash
command -v conda
command -v distrobox
command -v docker
nvidia-smi
python3 hope_training/whole_body_tracking/tests/test_table_tennis_geometry.py
bash -lc 'cd hope_training/whole_body_tracking && source setup_train_env.sh && hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"'
```

Results: the ignored A3 Isaac asset was restored locally from tracked Agibot materials and checked for
mesh references (`86` references, `0` missing). The host table-tennis geometry test passed (`6/6`;
torch aerodynamics skipped because host torch is unavailable). Later checks created `grasping` from
`docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` and verified it sees the RTX 5090 and `nvcc`.
`external_repos/IsaacLab` was cloned at tag `v2.1.0` / commit `21f7136`, and the local
`hope-isaac-py310` env now imports `torch 2.5.1+cu124`, `hydra 1.3.3`, `onnx 1.16.1`, and
`onnxscript 0.3.2` from inside `grasping`. `pip check` reports no broken requirements. Importing
Isaac/Kit reaches the NVIDIA Omniverse EULA prompt; no Isaac smoke test was run before explicit EULA
acceptance.

Additional local motion-environment checks from 2026-06-25:

```bash
cd hope_training/GMR
PYTHONNOUSERSITE=1 /home/agiuser/miniconda3/bin/conda run --no-capture-output \
  -n hope-motion-py310 python -c "import general_motion_retargeting, mujoco, smplx, torch"
```

Result: GMR commit `bb1bbe4` imports in `hope-motion-py310` with `torch 2.12.1+cu130` and
`mujoco 3.10.0`. GVHMR commit `6ec3ca3` is cloned but not installed because its requirements pin
CUDA 12.1-era `torch==2.3.0+cu121`/`pytorch3d` wheels that are not accepted for this RTX 5090 host
without a compatibility pass.

## Risks

- Training may fail before policy quality can be evaluated because of asset, observation, or reset issues.
- A weak first policy is still useful, but only if metrics and failure modes are recorded.
- Copying HITTER reward assumptions blindly may hide A3-specific limitations.
- TTRL can change upstream; record the source commit if it informs a training change.

## Next Steps

1. Re-run the unified `HOPEPingPongDeployParity` smoke and then the real unified forehand+backhand training run under the post-merge uniform-target defaults, from the corrected local `_hopex.npz` clips (`motion_file=`/`motion_file_2=`) or after uploading future verified registry artifacts and recording their aliases. Do not use the v5 R15 clips as product defaults.
2. Set measurable acceptance metrics for first usable baseline: fall rate, racket error at strike, physical recovery after clip wrap, and command latency assumptions.
3. Record exact local motion paths or registry artifacts, WandB run IDs when used, checkpoint paths, and ONNX export paths; evaluate the trained checkpoint/ONNX from the W&B run and record exact quality metrics and failure modes here.
4. Watch the exact-strike pass rates (`strike_pos/vel/normal_pass_exact`, `strike_composite_success_exact`) during long training under the uniform default; if a run opts into `target_mode=reference_perturbed`, also watch `ref_perturb_scale`, since that mode widens the target distribution only through its success-gated curriculum.
5. Run `scripts/sync_external_repos.sh` before using TTRL for comparison, and record the source commit for any extracted idea or config.

## Audit update 2026-07-10: formal training lineage

- Schema-v3 checkpoints/ONNX now bind instantiated action/joint order,
  decoder, PD, actuator integration, armature, effort/velocity limits, PhysX
  friction semantics, q-des limits, timing, observation/body layout, motion
  lineage and the exact racket control point. Legacy or override resumes
  cannot acquire an exact lineage merely by being re-exported.
- Motion kinematics schema 2 now declares rigid-point semantics and binds the
  complete articulation `body_names` column order. Isaac-compatible references
  store link-origin positions and COM-point linear velocities; old V5/MuJoCo
  files whose velocity is `d(link position)/dt` are rejected for formal use
  until explicitly migrated. Schema 1 is exact-ineligible because it lacked
  body order. `make_static_motion.py` also writes the full contract, because an
  all-zero clip cannot reveal its point semantics from content.
- Each clip FPS must be a finite positive scalar, every clip in a unified run
  must match, and the result must equal `1/env.step_dt`; schema-v3 records the
  per-clip FPS, full articulation order and selected-body index/name mapping.
- Actual racket speed uses the link-origin channel and target speed is derived
  from the same site-position path. `clean_reference_strike_velocity=false`
  is rejected rather than falling back to a COM/site mixture.
- A1 position/velocity/face/sign now traverse one atomic delay/drop message and
  reset clears held/drop/bias state. VirtualBall explicitly pins the intended
  `4/0.5/0.5` position/velocity/normal shaping and zero foot-orientation term;
  historical 10/5 and `-0.3` runs retain their saved config provenance.
- V5hLs target speed is still an `80 ms` (`+-2` frame at 50 Hz) average, not
  verified instantaneous contact truth. Contact frame and `+-1/+-2` window are
  preregistered ablations before a professional-transfer conclusion.
- Existing A3 checkpoints inherit uncalibrated PhysX joint-friction
  coefficients. The next formal wave needs zero-friction vs calibrated-
  friction controls; do not silently rewrite old checkpoints.

Fresh formal artifacts must be exported from the current schema and current
motion assets. Old ONNX files remain diagnostic only.

## Phase-1 local integration audit 2026-07-11

The local Phase-1 snapshot was preserved at
`codex/integrate-local-ablation-20260711@30f4652`, but its formal Isaac
evaluator was not merged.  It encodes question identity as packed
`(clip,row)` integers, consumes separate per-side sequential cursors and tries
to inject an exam bank into a command that the current schema-v3 stack
correctly constrains to the train split.  That conflicts with the production
content-addressed question IDs, immutable schedule and per-attempt seed.  With
many vector environments it can also exhaust a small side bank during one
reset.  These are protocol differences, not merge-conflict cosmetics.

The production MuJoCo BankExam remains the only **bookable** score path until
the new Isaac companion leg passes its runtime canary.  The companion adapter
is now implemented in `scripts/isaac_bank_exam.py`: it restores the saved
`env.pkl`/`agent.pkl`, verifies termination and checkpoint contracts before
`gym.make`, applies a content-addressed nominal evaluation profile, and assigns
one independently validated schema-v3 exam row to each environment through an
explicit runtime seam.  The saved training command and its train-split bank are
not replaced.  Both simulators can consume the same balanced schedule JSON,
content IDs, hold values and per-attempt action-noise seeds.  Its first runtime
acceptance test remains a fixed M3f/M2/G1 canary
of ten questions per side.  Every report cell uses a fixed question prefix;
if any prefix attempt is censored, the cell is invalid rather than filled by a
later question.  Only `noise_scale=0` survivors proceed to 50 questions per
side, continuous play, noise and a second seed.

The dependency-light review also closed four first-action/runtime hazards
before Pod launch: the nominal default-joint capture startup event is preserved
with randomization disabled; external injection refreshes actual racket FK and
strike timing before actor action 0; command resampling clocks are frozen; and
the shared hold contract is exactly `H` stand actions then raw frame 0. The
termination parser is schema v3 and records current hold-aware tracking guards
with `ignore_hold=true`. Actor observation histories greater than one sample
remain unsupported and fail before rollout rather than being double-advanced
by the release-frame refresh.

Three independent utilities were retained from the snapshot:

- `scripts/audit_runpod_terminal_runs.py` inventories historical terminal
  checkpoints and prints judge commands without executing them;
- `scripts/termination_contract.py` freezes timing and termination semantics
  from a saved run and is now checked before Isaac environment construction;
- `scripts/virtual_return_scorer.py` specifies the Isaac 10 ms RK4 scorer and
  ball-centre contact plane; MuJoCo now delegates its actual and counterfactual
  return scores to this implementation without changing the physics-hash-bound
  `venue_ball_sampler.py`.

The shared schedule and adapter tests are reproducible from
`docs/operations/build_and_test.md`. Local verification on 2026-07-11 passed
`67` adapter/audit tests with one optional Torch parity skip, `85` formal CPU
contract tests, and `141` unique tests in their combined contract run with the
same one optional skip. M2's Isaac leg is now verified below; Pod M3f/G1 and
the companion same-paper MuJoCo legs are still required, so this gate remains
`Partial` and historical checkpoints remain diagnostic-only.

The first M2 q1/side Kit smoke on Pod1 reached `gym.make` and then correctly
failed on the current MotionLoader's link-origin-vs-COM guard: the historical
v4rg `_cal` files are untagged and their `body_lin_vel_w` is numerically the
finite difference of link-origin position. The exact path remains unchanged
and still refuses those files. The historical evaluator now has a separate
default-off `allow_legacy_link_origin_velocity` seam, enabled only after the
explicit inexact preflight detects that content signature; the profile records
motion paths/SHA and the resulting legacy command semantics. This preserves
the checkpoint's old input meaning for ruler diagnosis without making the
motion eligible for fresh training or a bookable score. The next retry passed
the motion and legacy-bank loaders, then found a normal Python pickle-evolution
gap: the old `RacketTargetCommandCfg` predates `rally_legacy_metrics`. The
inexact path now fills only dataclass/configclass fields that have an explicit
current default and records every filled field; exact evaluation refuses any
such hydration. The subsequent retries are recorded below.

That rerun reached complete environment and policy construction before the
historical checkpoint exposed four zero observation-normalizer `_std` entries.
This is valid for the saved rsl_rl implementation because inference uses
`(x - mean) / (std + eps)` and the configured `eps` is `1e-2`. The inference-
only compatibility loader now accepts finite non-negative std values only when
epsilon is finite and positive; it continues to reject negative/non-finite
scales, zero/missing epsilon, dimension mismatch and missing actor state. This
is covered by a CPU regression. The next two retries below close the remaining
writer issue and regenerate the same q1 cell.

The next identical-paper retry completed both Isaac attempts and failed only
while assembling source provenance: a positional `Path.parents` index treated
`.../hope_training` as the checkout root, producing a duplicated
`hope_training/hope_training/.../virtual_ball.py` path. Repository discovery
now uses the checkout's venue-physics and whole-body-tree markers, and a CPU
test requires both hashed source paths to exist. The scorecard must still be
regenerated; no partial in-memory result from the failed writer is accepted.

Retry 4 regenerated the q1 cell successfully at commit `a619aa4`. The result is
valid and uncensored, with the exact bank SHA, schedule SHA
`7809555811788675a26705deb9495159210c6f449b17aeb96161d73ecc34160a`
and ordered question IDs from the supplied paper. Both `hold_steps=0` attempts
were retained as guard-reset failures (0/2); nothing was replaced. The
quota-10 paper then completed all 20 attempts with deterministic nonzero holds,
20/20 exact reaches and hits, no falls/guards/censoring, and 16/20 returns
(forehand 6/10, backhand 10/10). The JSON SHA is
`e625a09c31931a5c4cdcd8118f96ddc351b9eb0f2cad59cf265f26107eb787fc`.
This is successful runtime acceptance evidence for the historical M2 Isaac
leg, but `evaluation_contract_exact=false`; M3f/G1 and the same-paper MuJoCo
legs remain required before this gate can move beyond `Partial`.

M2 then advanced to the fixed 50-per-side slice. The first execution was
discarded because a concurrent task fast-forwarded the checkout during the
cell, so its end-of-run source hashes could not prove the code that had been
loaded. The rerun held `c69ff13` fixed and produced 100 finalized uncensored
rows with the supplied schedule SHA
`9d1a1d601b098f93ab151a9d9dfabf6a92a81b4b2896230bfebf7483e20324cd`:
100/100 exact reaches and hits, 86/100 returns (forehand 36/50, backhand 50/50),
no fall and no guard reset. Valid JSON SHA:
`723322b469b105282506fd7b2536e79bf6cd24e338cecd50536296f773015a01`.

M3f and G1 completed the same quota-10 paper contract at fixed checkout
`c69ff13`: M3f returned 20/20, while G1 returned 10/20 with backhand 0/10.
Both had 20/20 exact reaches/hits, no physical falls or guards, complete
uncensored ledgers, and exact bank/schedule/order equality. G1 stopped at the
known-bad canary. M3f's 50/side clean result was 99/100 returns with one
zero-hold forehand tracking guard; 5% action noise remained 99/100 and schedule
seed 1 was 100/100. M2 was 86/100 clean, 85/100 at 5% noise and 91/100 on
schedule seed 1. Full artifacts and hashes are recorded in
`docs/PHASE1_SCHEMA3_RESULTS_2026-07-11.md`. These cells validate the runtime
adapter but remain historical/inexact and cannot close this gate.

### Phase-1 main-matrix closure and fresh-lineage preflight (2026-07-11)

The historical main matrix now uses the accepted same-paper adapter rather
than old scorecards. R1b (two policy seeds), R5b (two policy seeds), C1, G2
and M2f3 completed ten clean questions per side in both simulators. R5b seed
1/2, G2 and M2f3 stopped because MuJoCo forehand returned 0/10. R1b reached
q50 but both policies collapsed to 3/50 MuJoCo forehand returns, versus 45/50
and 40/50 in Isaac. C1 completed q50 clean, noise and second-schedule cells:
Isaac clean was 46/50 forehand and 50/50 backhand; MuJoCo clean was 40/50 and
10/50. All ledgers were complete and uncensored. The full tables and hashes
are in `docs/PHASE1_SCHEMA3_RESULTS_2026-07-11.md`.

C1 is therefore an additional historical diagnostic survivor, not a formal
winner. M3f remains stronger in MuJoCo and carry-state continuity. Every
checkpoint above still lacks schema-v3 training lineage and records
`evaluation_contract_exact=false`.

Fresh training also needs an explicit plant control. The default A3 config
continues to preserve its historical, uncalibrated PhysX joint-friction
coefficients. A new default-off `task.plant.zero_joint_friction=true` override
zeros every actuator before `gym.make`, so the saved env and schema-v3
training contract bind the actual all-zero vector that formal MuJoCo can
reproduce. False/absent is a no-op, unknown keys and malformed booleans fail
loud, and legacy warm-starts remain exact-ineligible. An isolated Pod worktree
passed the override and schema-3 contract tests (`60 passed`); Hydra composition
accepted the declared leaf and rejected a misspelled parent. The true Kit
smoke additionally requires the post-`gym.make` 31/31-zero assertion to pass.
The fresh
migrated-motion/bank runtime smoke and training launch are still required, so
this gate remains `Partial`.

The motion preflight subsequently found and fixed a real body-column mismatch.
The old source order cannot be reused as the live Isaac articulation order;
the migration now takes `--target-body-order`, reorders all four body arrays by
name, and only then converts link-origin velocity to COM velocity. The first
incorrect outputs are quarantined. The corrected v4rg pair is schema 2 at
50 Hz with the tracked 32-body runtime order. Its schema-v3 train/exam banks
share family `b21c161a...28ad5`, have disjoint question IDs and passed every
strict Torch/physics/loader check. Exact paths, counts and full hashes are in
`docs/PHASE1_FRESH_LINEAGE_2026-07-11.md` and the tracked asset manifest.

Pre-launch contract review also closed a diagnostic-export trap: a legacy
continuation still writes a complete schema-3 execution sidecar, but its
inexact motion must not be rejected before it can produce a diagnostic ONNX.
The exporter now requires structural validation plus the checkpoint/sidecar
SHA binding and emits `training_contract_exact=0`; a checkpoint claiming exact
lineage still goes through the stronger schema-2 motion gate. The hard contract
also binds whether face command is enabled and which face pairing is selected.
`scripts/launch_phase1_20260711.sh` hashes every parent/motion/bank and pins the
matched M3/M2 controls plus two fresh seeds. The full 179-D Kit construction
smoke has now passed with hard-contract SHA `3a3b3d95...b9972`. All four
causal continuations and both fresh seeds reached their first PPO iteration at
checkout `6d93bcb`; first checkpoints bind the matching schema-3 sidecar, with
inexact lineage for resumes and exact lineage for fresh runs. Within each M3
or M2 pair, recursive contract diff reports only `face_command_pairing`.
Training is still running and no terminal checkpoint or score exists, so G05
remains `Partial`.

### Phase-1 breadth and checkpoint-curve correction (2026-07-11)

The first launch occupied six GPUs but ran only one 4096-env process per GPU.
That is not the established breadth policy. Historical measured operation is
four jobs per 5090 (about 22--23 GiB total), with serialized Kit boots and a
75-second stagger. The reviewed scale-out is now 24 arms: a second paired
continuation seed for each M3/M2 old-vs-S1 family plus a four-seed fresh 2x2
factorial over face pairing and zero/non-zero plant. Only the fresh
`shared_plus_y + zero-friction` (`SZ`) cell is the pre-registered formal target;
the other cells are causal diagnostics. The exact assignment is tracked in
`configs/phase1_scaleout_matrix_20260711.json`.

Waiting for the terminal checkpoint was also contrary to the recorded
checkpoint policy. The first curve workers target causal `17000/18000/19000`
and fresh `0/1000/2000` checkpoints, with one Isaac export at a time per Pod
and CPU MuJoCo exams allowed to overlap. Their first preflight found a missing
ignored A3 asset link in each detached worktree; after linking the frozen
training asset, a second preflight wrote ONNX but exposed a buffered/missing
success handshake. The next preflight reached the normalizer sidecar and found
four saved `_std=0` constant features. Runtime already evaluates them with
`eps=0.01`; the writer now matches that contract (`std>=0`, `std+eps>0`) while
negative/non-finite values remain fatal. All failed batches are retained and
none is a checkpoint score. `judge.sh` now forces unbuffered export output for the retry. Later points follow
the 1000--2000 iteration schedule and densify around a measured peak. The
worker records checkpoint/evaluator hashes and never signals a training
process. Both scale-out roles (three layers of three arms on each Pod) and all
18 initial checkpoint jobs passed dry-run input/hash/path checks. Actual curve
results, the corrected retry and the remaining 18 first-iteration contracts are still pending, so
the gate remains `Partial`.
