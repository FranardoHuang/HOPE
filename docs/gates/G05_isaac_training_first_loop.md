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

Follow-up note (2026-07-13, seed-budget correction; Gate remains `Partial`):

- The rejected `SZ` family used four from-scratch seeds through model-4000. That was enough to
  establish instability and expose a signed-face measurement defect, but continuing to replicate a
  rejected recipe bought more evidence than the baseline decision needed.
- Future mechanism screens use one blocking seed first. One 5090's four breadth slots hold four
  distinct causal cells, with relative checkpoints at `+200/+500/+1000`; only a surviving cell together with its
  matched control receives a second seed. Three to four seeds and terminal training are reserved for
  a candidate that could actually become the accepted baseline.
- The first application is the proposed hot-start/fresh × face-guidance-off/on signed-face funnel.
  It has no source SHA, machine prereg or Pod run yet, so this note does not authorize launch and does
  not add accepted training evidence. See
  [the experiment record](../experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md) and
  [the acceleration policy](../research/phase1_ablation_acceleration_2026-07-11.md#seed-是晋级税不是首轮并发单位).

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

1. Hand the best exact fresh `SZ model_2000` checkpoint and its 179-D contract to G06's native
   MuJoCo training/fine-tune P0; do not substitute another Isaac-only sweep for that handoff.
2. Re-run the unified `HOPEPingPongDeployParity` smoke and then the real unified forehand+backhand training run under the post-merge uniform-target defaults, from the corrected local `_hopex.npz` clips (`motion_file=`/`motion_file_2=`) or after uploading future verified registry artifacts and recording their aliases. Do not use the v5 R15 clips as product defaults.
3. Set measurable acceptance metrics for first usable baseline: fall rate, racket error at strike, physical recovery after clip wrap, and command latency assumptions.
4. Record exact local motion paths or registry artifacts, WandB run IDs when used, checkpoint paths, and ONNX export paths; evaluate the trained checkpoint/ONNX from the W&B run and record exact quality metrics and failure modes here.
5. Watch the exact-strike pass rates (`strike_pos/vel/normal_pass_exact`, `strike_composite_success_exact`) during long training under the uniform default; if a run opts into `target_mode=reference_perturbed`, also watch `ref_perturb_scale`, since that mode widens the target distribution only through its success-gated curriculum.
6. Run `scripts/sync_external_repos.sh` before using TTRL for comparison, and record the source commit for any extracted idea or config.

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

The following retries reached the CPU evaluator and separated two more evaluator faults from
training quality. Both Pod venvs had `onnxruntime` but not the `onnx` package required for formal
graph checking; they now pin `onnx==1.22.0`, and checker plus runtime inference pass. Fresh exact
checkpoints then failed before rollout on only `2.71e-9` armature disagreement caused by float32
metadata versus float64 MJCF parsing. A later retry found the analogous `3.0517578e-6` residue at
the `118.2` ankle effort limit. The formal gate now compares exact float32 grid identity instead of
using a fixed tolerance, with midpoint/next-grid regressions that reject material plant differences. The six
original trainers remained live and the frozen checkouts stayed clean. None of these evaluator
failures is counted as a model result, and G05 remains `Partial` pending the corrected fresh curve
and layer-by-layer scale-out proof.

The scale-out proof is now complete, although training/results are not. Both Pods reached four
4096-env trainers on each of their three RTX 5090s. The full-pool snapshot used
`22.9--23.2/32.6 GiB` per card at `87--97%` utilization, with `840/904 GiB` host RAM still
available. All 24 accepted arms reached a first PPO iteration; every first checkpoint is finite and
its embedded contract SHA matches the adjacent schema-3 sidecar. Pod 1 LZ seed 3 had one
pre-contract scene-start `malloc` abort; its log/launch-state SHAs are retained, the process exited
itself, and an unchanged single-arm retry (PGID `1354525`) passed. The failed boot is not a 25th
experiment. G05 remains `Partial` because periodic curves and terminal verification are incomplete.

The first corrected formal curve is now real training evidence rather than an evaluator preflight.
At clean q10 on the same immutable paper, fresh `SZ` seed 1 returned
`0.00 -> 0.50 -> 0.90` at checkpoint `0/1000/2000`, and seed 2 returned
`0.00 -> 0.50 -> 1.00`. All six jobs completed `rc=0` with exact schema-v3
evaluation. This proves why checkpoints are tested during training, but the
10-per-side prefix remains direction-only and cannot authorize stop/promotion.
The original causal 20000 screens also completed: M3 old/S1 was `0.45/1.00`,
while M2 old/S1 was `0.50/0.50`; causal results remain inexact diagnostics.

Periodic coverage now extends to all 18 newly launched arms. Deterministically
generated per-Pod causal/fresh manifests cover 142 additional clean q10 jobs,
with milestone-major barriers and separate queues so a causal terminal does not
block a fresh 2000-point screen. Generation and current worker compatibility
pass locally (`7 passed` including the venue timing and existing worker tests).

A separate audit found that the live pool is continuous only in the slow
complete-clip sense. The bound motions plus hold produce same-player strike
intervals around `2.90/3.75/4.60 s` (q10/median/q90), versus a conservative
venue A-B-A sample at `1.757/1.903/3.356 s`. The next target currently appears
only at clip wrap, later than the measured opponent-hit event. Therefore these
24 arms cannot satisfy the arbitrary-time continuous-play acceptance item.
They remain unchanged; an event-driven `T0/T1` timing pair, longer/opportunity-
count episode and new hard-contract timing fields are specified in
`docs/research/phase1_continuous_rally_timing_2026-07-11.md`. G05 remains
`Partial` pending terminal/paired q50 evidence and that separate continuity lane.

### New motion-library intake, TOPP and recovery lane (2026-07-11)

Ten private Franco/v6/v7 air-swing videos are now registered by the tracked,
content-addressed `configs/motion_video_intake_20260711.json`. Local and Pod1
copies passed byte/hash/media validation. `scripts/audit_motion_video_intake.py`
the structural `scripts/audit_gvhmr_result.py`, and the memory-gated
byte-exact gzip source archive `docs/experiments/archive/run_motion_video_gvhmr_queue_20260711.py.gz`
and the tracked result binding are covered by dependency-light
tests. Pod1 queue PID/PGID `1383735` completed all ten reconstructions after
GPU1 naturally crossed the `19000 MiB` launch gate. It processed the Franco
forehand-block item first, then the remaining Franco and v6/v7 clips, and was pinned to GVHMR `6ec3ca3`, a
clean worktree, the full checkpoint/body-model tree and motion-Python freeze;
each result matches the source frame count and finite SMPL tensor shapes.
The queue ran outside the frozen Phase-1 checkout. Its queue-state SHA and all
ten result/audit hashes are tracked in
`configs/motion_video_gvhmr_results_20260711.json`.

The next CPU-only stage is also complete as a **diagnostic**, not as a motion
promotion. A repo-owned serial GMR queue required a clean source checkout at
`aabea2eee4be4bc16d4be17dac5ffa85e5a31539` plus a verified recovery bundle,
kept frame-zero warm-up enabled, and retargeted all 10/10 items to finite
30 Hz, 31-DoF A3 pickles in 52 s. Every item converged below
`max|dq| < 1e-4`; the exact source/output/log/audit/tool/environment bindings
are in `configs/motion_video_gmr_results_20260711.json`. These outputs retain
per-video GVHMR body betas and therefore carry
`body_shape_contract=diagnostic_video_betas` and
`formal_eligible=false` by construction.

A deeper read-only replay of the Franco forehand-block pilot found useful and
blocking evidence at once. Joint order matches the canonical 31-joint MJCF,
all joint samples are inside limits, 30 Hz finite-difference speed stays below
the URDF limits, and 641 sampled/interpolated canonical-MuJoCo poses reported
zero robot self-contact. That finite-substep test is not a continuous
collision certificate and the MJCF lacks table/net geometry. More
importantly, all 65 frames penetrate the floor, with the lowest collision geom
roughly `7.7--8.4 cm` below zero. The current root trajectory is therefore
blocked on ground/root calibration followed by repeat collision, dynamics and
table/net clearance gates. A repo-owned `scripts/ground_gmr_pkl.py` now
implements the first step without directory scans or in-place edits: it binds
the exact input and canonical MJCF SHAs, computes the lowest world-z support
over enabled robot collision geoms at each source frame, applies one constant
root-z translation, and emits a no-clobber pickle plus SHA-bound report. Real
canonical-MJCF mesh tests pass. Pod1 then ran the tool no-clobber on all ten
diagnostic GMR outputs. Original discrete-frame minima were
`-0.08072..-0.08716 m`; every per-motion fixed root-z shift left a global
minimum of about `10 um`. The ledger
`configs/motion_video_gmr_ground_results_20260711.json` binds all ten
input/output/report SHAs to tool `db5bd167...`, canonical MJCF
`2ab1cd31...` and compiled collision digest `18e7f6ff...`. These remain
per-video-betas diagnostics: inter-frame ground/self collision, dynamics,
table/net clearance, canonical betas and all later gates remain open, so none
is eligible for schema-2 promotion or RL yet.

The upstream body-shape normalization is now materialized separately. A
CPU-only no-clobber tool gives each of the ten videos one equal vote and writes
one shared 10-D beta vector (SHA `a03f1642...9cc6`) into ten new GVHMR PTs;
all non-beta semantic digests remain bit-exact after save/reload. The result is
bound in `configs/motion_video_canonical_betas_result_20260711.json`. There is
no measured performer height: GMR's `1.73066 m` value is explicitly only its
beta heuristic, so the artifacts remain diagnostic, non-calibrated and
formal-ineligible. The historical preregistration's unbound guess that the
loader padded six zeros has been revoked: clean GMR `aabea2e` loader SHA
`2737f472...5de2` actually selects
`betas[0].detach().cpu().numpy()[:10]` with no padding.

A body-shape-aware CPU-only no-clobber queue then retargeted all ten
canonical-beta PTs in 48.7 s (PID/PGID `1442090`). All outputs are 30 Hz,
31-DoF and finite; frame-zero warm-up converged in 16--29 rounds with final
max `|dq|=6.88e-5..9.76e-5`. GMR remained clean and no GPU was allocated.
Exact source/output/log/audit bindings are in
`configs/motion_video_canonical_gmr_results_20260711.json`. These remain
diagnostic: the ten new outputs still need independent no-clobber grounding
and dense collision, racket/handle-to-body, dynamics and table/net gates.

The canonical-GMR grounding prerequisite has since completed independently for
all ten inputs. Its content-addressed ledger is
`configs/motion_video_canonical_gmr_ground_results_20260711.json`; every
30 Hz source-row minimum is about `10 um` above the bound floor after one
per-asset constant root-z translation. A separate CPU-only screen then replayed
654 source rows as 5,162 finite samples at 240 Hz. It found zero ground-danger
samples, zero robot self-interpenetrations, zero racket/handle-to-critical-body
samples below the 5 mm hard threshold, and zero below the 20 mm warning
threshold. The smallest observed body clearance was `40.2466 mm` in
`franco_backhand_loop_a`. This is finite dense sampling, not a mathematical
continuous-time certificate, and the canonical MJCF still has no table/net
collision geometry.

The same screen extracts the official vendor-MJCF `right_racket` site centre
and local +Y face plus `mj_differentiatePos`/`mj_objectVelocity` site speed.
It deliberately does **not** report a hit phase or motion-library coverage yet.
Canonical grounding changed root z only; it did not bind GMR world to the HOPE
+X virtual-table frame, and intake mirror status remains unverified. A first v2
result that scored venue questions anyway is preserved but all of its return,
phase, selector and two-vs-four fields are revoked; only its safety subtree is
accepted, and that subtree equals v3/v4 for all ten inputs. Accepted v4 freezes a
64-question paper but records `consumed_for_returnability=false`, with every
phase/coverage/selector field null and blocked. The preregistration and compact
ledger are `configs/motion_video_gmr_phase_safety_prereg_20260711.json` and
`configs/motion_video_gmr_phase_safety_results_20260711.json`. Schema-2 plus
HOPE +X reground (or an independently verified proper-rigid transform) and mirror
semantics must land before that paper can run. No motion entered RL or hardware.

That diagnostic frame prerequisite has now passed without claiming a capture-table
extrinsic.  Ten content-bound midpoint crops show upright, unreflected Chinese
background labels; independently, every canonical GMR is right-arm dominant by
at least `9.98x` versus the left arm (`5x` preregistered threshold).  Each clip's
proper-rigid matrix is derived only from frame-0 pelvis XY/heading and the audited
ground plane, mapping the root to the HOPE origin/+X while preserving z.  The
matrix set was frozen before scoring.  The target is explicitly the standard
counterfactual HOPE virtual table, not the room in which the air swings were
recorded.  Evidence is in
`configs/motion_video_gmr_frame_contract_results_20260711.json`.

The v5 CPU runtime then consumed the unchanged 64-question paper (result SHA
`c299b7a0...`) and reproduced every v4 dense-safety subtree.  Exact zero-retarget
coverage is `0/64` for all motions and libraries, with zero common support, so it
cannot choose two versus four actions and must not be paraphrased as "all motions
are ineffective."  Intrinsic relocation-only evidence retains Franco backhand
loop B (`32/32`, phase `0.5444`) and C (`27/32`, phase `0.5155`) as spatial-retarget
candidates; A is `1/32`, all others zero.  None is TOPP-eligible yet.  TOPP remains
paused until explicit spatial retarget, schema-2/L0/L1, table/net and dynamics
gates; final motion/library acceptance belongs to AgiBot vendor MuJoCo
Gate3 runtime/stability first and Gate3B no-reset behavior scoring second.  Compact ledger:
`configs/motion_video_gmr_phase_counterfactual_results_20260711.json`.

The 2026-07-13 interpretation is stricter: motion effectiveness is the motion's
own safe contact-time manifold crossed with a compatible incoming-ball/stroke
question family and a legal whole-trajectory `SE(2)` stance. B (`frame 49`,
`32/32`, nearest old question `0.165 m`) and C (`frame 50`, `27/32`, `0.237 m`)
fit only the `0.30 m` translation-norm bound and remain candidates. Forehand waits on
the roughly `170 deg` face-sign ambiguity, and block motions need a block-specific
paper. Schema-2/L0/vendor-L1 self-hit/full table-net swept clearance `>=5 mm`
grants training eligibility, not evidence of return effectiveness.

The next spatial step is now preregistered and mechanically checked, but has
not been promoted or run against restored private evidence.  Plan
`configs/motion_video_spatial_retarget_prereg_20260712.json` (SHA
`d8c918ac...5a9f`) keeps all ten motions on every matching immutable question;
the B/C intrinsic result affects ranking only.  R0 permits translation only;
R1 permits the frozen yaw grid `[-10,-5,0,5,10] deg` plus translation.  Each is
one ground-preserving proper SE(2) transform applied atomically to the entire
motion: no z, scale, reflection, joint or per-frame edit, and no capture-table
extrinsic claim.  The station envelope is norm `0.30 m`, `|x|<=0.20 m`,
`|y|<=0.30 m`.  The CPU tool/test contract passes `7` tests and rejects skipped
assets, unsafe/wrong-side frames, out-of-envelope stations, clobbering and
incomplete certificates.  The laptop lacks the exact 792,241-byte full v5
result, and the current manifest deliberately records
`certificate_bundle_preregistered=false`; therefore it can produce only
proposals after exact evidence restore.  Promotion requires candidate-bound
runtime-order schema-2 materialization, L0 PASS, vendor-MJCF L1 PASS and a
whole-trajectory table/net swept-clearance PASS with at least `5 mm` margin.
Dynamics/balance and TOPP remain downstream; Gate3 runtime/stability must precede Gate3B no-reset
behavior scoring, and RL/hardware remain blocked.
No GPU, Pod, trainer or hardware was touched.  Reproduction and the restore
boundary are in `docs/operations/run_motion_spatial_retarget_screen.md`.

This is not yet a training result. The videos contain no ball/table/contact
truth; final-pixel mirror status is now verified but monocular depth/capture-table
extrinsic remain unverified, and the three Franco
backhand-loop recordings are candidates for one semantic action rather than
three new action classes. Every candidate must pass A3 schema-2 conversion,
finite/limit/endpoint checks, vendor-MJCF self-collision and racket/handle-to-
body/table/net swept-clearance gates before returnability phase scanning.
Native/TOPP v3 assets are then compared on the same spatial path and strike
constraints, with both outputs re-audited.

Formal two-vs-four-action training is blocked on a dynamic clip catalog and a
shared global-question axis: current upper layers still encode two clips and
sample clip before question. The preregistered comparison therefore has two
separate fairness papers, equal total transitions and equal per-action
exposure, plus common-action non-regression and a train-fitted frozen stable-
action selector. Between-shot work uses strike/absorb/recover/ready states and
a tolerant ready set rather than a dense reward back to an exact frame 0.
Repository Hitter evidence warns that post-strike brake/ready rewards can
propagate through GAE and damage the hit; recovery is first isolated as an
option/bridge, then evaluated on event-driven T0/T1 timing. Full rationale,
literature and stop/promote gates are in
`docs/research/motion_library_topp_recovery_2026-07-11.md`. G05 remains
`Partial`; no new motion is authorized for hardware.

The first real continuation terminal also corrected a queue-index bug. Pod2
M2-S1 exited normally with `model_20998.pt` (`iter=20998`), not 20999; all
1,762,715 floating checkpoint elements are finite, checkpoint SHA is
`574ff640...0049`, and embedded/adjacent contract SHA is
`7268eb38...28f2`. Its schema-3 legacy-motion lineage is correctly inexact.
The cadence and scale-out causal manifests now use 20998, with a generator
regression. Only the affected waiting-worker PGIDs were replaced; fresh
scale-out workers and training arms received no signal. After the later split
and global hardening transaction, the six current original/scale-out worker
PGIDs are Pod1 `1432280/1432292/1432304` and Pod2
`200706/200718/200730`. Only their recorded legacy worker PGIDs received
TERM; trainers and judges received no signal. Five available old states were
rejudged rc=0 under manifest/job/job-contract bindings rather than silently
reused. The full transaction ledger is
`configs/phase1_global_curve_worker_hardening_result_20260711.json`.

Pod1 M3-S1 has now reached the same terminal integrity point. Its accepted
`model_20998.pt` has SHA-256
`a924048810aebda864bbf1f7b156ef4c4aa2c60ec4d65da6ae8977833deaa21e`,
contains `iter=20998`, and has zero non-finite values across 1,762,715 floating
elements. The embedded schema-3 contract SHA matches the adjacent contract
(`d3ff715e...29d9ce`), and the run log contains the contiguous 4,000 records
`16999..20998` with no NaN/Inf/traceback/OOM/malloc/killed signature. The
launcher did not persist an OS exit code, so the terminal checkpoint,
contiguous log, absent process and zero error signatures are the recorded
completion evidence. Its legacy-motion parent keeps this result causal and
inexact. M3-old has since naturally produced its own finite
`model_20998.pt` and exited. Its checkpoint SHA is `320b77c9...417a`,
embedded/adjacent contract SHA is `7542c59b...d941b`, and lineage remains
causal/inexact; the complete audit is
`configs/phase1_M3_old_terminal_audit_20260711.json`. The paired immutable
terminal q10 then completed on schedule `7a908142...d614`: M3-old
FH/BH/aggregate=`0.50/0.40/0.45`, M3-S1=`1.00/1.00/1.00`, aggregate delta
`+0.55`. This is a 10-per-side direction screen, not a stop/promotion
decision. Its ledger is `configs/phase1_M3_terminal_q10_pair_20260711.json`.

The triggered shared-schedule MuJoCo q50 has since completed. Both terminal
checkpoints consumed the same K=100 schedule semantic SHA
`949eb196...8fc0`, 50 attempts per side, seed 0, no noise and no censored
attempts. M3-old returned FH/BH/aggregate `31/50,11/50,42/100` and contacted
`89/100`. The raw ledger resolves the summary's legacy `fell=9` union into
**one physical fall plus eight non-physical guard resets**, not nine physical
falls. M3-S1 returned `50/50,50/50,100/100`, contacted `100/100` and had zero
such terminations. The aggregate paired delta is `+0.58`.
This selects M3-S1 only inside this legacy swing-family causal diagnostic; both
lineages and evaluations remain inexact and no formal/deployment/hardware
promotion follows. The first runner attempt judged M3-old but rejected its
own wrong result-schema assumption before starting S1; that attempt is
preserved. The corrected v2 reproduced the identical schedule bytes and reran
both cells successfully. Full hashes and limitations are in
`configs/phase1_M3_terminal_q50_result_20260711.json`. The same-paper Isaac
companion then scored both old and S1 at FH/BH/aggregate
`0.98/1.00/0.99`, delta zero. It does not reproduce MuJoCo's `+0.58`
ranking, so the cross-engine causal gate stays open and S1 can be selected
only inside that MuJoCo family/evaluator. Full companion hashes are in
`configs/phase1_M3_terminal_q50_isaac_result_20260711.json`.

The question-level forensic in
`docs/research/phase1_cross_engine_saturation_forensic_2026-07-11.md` further
shows that the two runs did not yet share an outcome instrument. Fresh model
4000's FH racket-center error is `13.15 cm` in MuJoCo versus `2.48 cm` in
Isaac, and Isaac's analytic face orientation erases M3-old's signed BH face
error (`168.15 deg` before orientation). The cross-engine gate therefore stays
open. A fail-closed 2x2 instrument-parity prereg now requires both physical and
analytic cells from both engines without changing the frozen thresholds; its
current blocker is missing Isaac post-contact physical truth.

The independent continuous-timing axis is now content-addressed but remains
launch-blocked. Its venue aggregate binds raw strikes SHA `6ad3c459...` and
records the overlapping-window, 16/21 high-ball and 2.5 s right-censor limits;
`1.903 s` is not used as a target. T0/T1 instead share a balanced engineering
event grid and freeze motion/TOPP/plant/face/reward/2-vs-4. Design validation
passes; launch validation must fail until the post-strike scheduler,
materialized schedules, continuous Isaac/MuJoCo judges, self-hit instrument,
fresh exact checkpoint and semantics-correct plant are all bound. Prereg SHA
is `2e7c4a34...2289c`; no timing arm has launched.

The training-side core is now implemented at `be5d7cf`: only an accepted exact
strike can arm the absolute event clock; reveal atomically installs the bank
row/native clip/exact hold/fixed deadline, and misses or infeasible rows still
consume the original opportunity without teleport/reset/history-noise reset.
Every timing-changing field is in the hard contract. This does not mutate the
frozen prereg or authorize launch; materialized schedules, continuous judges,
self-hit instrumentation, fresh baseline and calibrated plant remain open.

Natural terminal release also opens a separate second-wave causal paper rather
than permission to mutate the frozen 24-arm matrix. The preregistration
`configs/phase1_causal_followups_20260711.json` completes two
old-helper/S1-only/S1+guidance triangles with four arms: M3 gets missing
S1-only guidance-0 seed 1/2; M2 gets missing S1+guidance-`-0.95` seed 1/2.
All remain 4,000-update causal/inexact continuations. Their external
content-addressed launcher refuses dirty/wrong train or eval checkouts,
wrong assets/tools, a fourth occupied GPU slot, reused run names or a live
M3-old predecessor; it verifies each emitted hard-contract before starting an
independent `17000/18000/19000/20000/20998` q10 worker. The original 16999
parent is never copied beside the new sidecar or judged as the new contract.
q10 remains direction-only and q50 is an inactive separate template. After a
read-only duplicate-PID fix and a second clean validation, all four followups
launched without signalling an existing trainer. Accepted trainer/worker PGIDs
are Pod1 M3 seed1 `1409914/1410648`, M3 seed2 `1411167/1412047`; Pod2 M2
seed1 `196177/196753`, M2 seed2 `197146/197939`. All four emitted the expected
family hard-contract SHA, reached 17000 with zero logged bad signatures, and
restored the full pool to four trainers per GPU (24 live). Their first q10
screens were M3 S1-only seed1/2 `0.60/0.55` aggregate and M2 S1+guidance
seed1/2 `0.30/0.30`; these are inexact 10-per-side direction points only.

The first adjacent scale-out pair reinforces that rule. On identical K20
schedule `75aca567...51d7`, M2 seed2 old/S1 changed from `.40/.60` at 18k to
`.50/.40` at 19k, reversing the small-paper order. Both 19k checkpoints are
finite and bind iteration, adjacent hard-contract SHA and causal lineage.
They continue unchanged; no stop, promotion or q50 trigger follows. The
content-addressed curve ledger is
`configs/phase1_M2_seed2_18k_19k_q10_curve_result_20260711.json`.

The first 17k run also exposed a provenance gap before later milestones: the
launcher had deliberately pinned eval checkout `46a0ce2`, whose worker SHA
`8b980359...` predates the checked-in screen-policy/job-contract state fields.
The four judge commands and rc=0 results match the immutable manifests, but
their state JSON lacks manifest/job/job-contract SHAs. A separate replacement
contract now requires both workers on one Pod to be alive, exact-PGID,
childless and manifest-bound before any signal; it then TERM-signals only those
workers, preserves the legacy evidence, starts standalone hardened worker
`21e30153...` with a fresh state dir, and rejudges 17k before accepting the
correction. Trainers/judges are out of scope. Both Pod transactions have now
completed. Old childless workers were TERM-signalled only at the four exact
PGIDs above; hardened worker PGIDs are Pod1 `1416771/1416784` and Pod2
`198759/198771`. The correction-sidecar SHAs are respectively
`2faf88de...ffe3`, `1d6f8ba3...bae9`, `0dd02fae...d165` and
`45f4334d...0ad`. Rejudged 17k states returned rc=0 and bind manifest, job
spec and job contract SHAs plus checkpoint, judge, clean training `6d93bcb...`
and eval `46a0ce2...`; legacy state/log bytes remain preserved.

`SZ`'s target label is now explicitly scoped: it is the only current fresh
cell whose zero-friction plant can be replayed with the same schema-v3
cross-engine execution semantics. It is **not** a deployment-plant or hardware
candidate. The 2026-07-07 frozen probe already showed a zero-friction policy
degrading from virtual hit `0.9997` to `0.63` and fall `0.27` to `0.87` when
moved into the non-zero-friction plant. `SP/LP` do not resolve this: their
PhysX values are the historical unit-mismatched copy of MuJoCo constant-Nm
`frictionloss`, not calibrated friction. A new from-scratch `SC` cell requires
measured friction semantics plus separate PhysX/MuJoCo adapters and hard-
contract hashes before deployment-plant, continuous-practical or Gate3B
promotion. This does not block the same-schedule SZ q50 required for current
execution-contract model selection. `SP` is explicitly judged inexact so its
non-zero plant cannot fail the formal profile and block later SZ milestones.
The current 24-arm training recipes remain unchanged. The separately
validated repair preregistration is in
`docs/research/phase1_plant_semantics_repair_2026-07-11.md` and
`configs/phase1_plant_semantics_repair_prereg_20260711.json`; it is currently
`blocked_on_calibration_evidence`.

The 2026-07-12 provenance recheck corrected that v1 manifest's declared
repository snapshot from training-only ancestor `612f54d` to `d4ca566`, the
first commit containing all eight already-recorded source hashes. Current main
has since changed `training_contract.py` for strict face179, so current-checkout
verification now deliberately returns exit 2. This is an additional fail-closed
prelaunch blocker: a new reviewed preregistration must bind current source bytes
before any `SC` arm. It does not change the running `SZ/SP/LZ/LP` recipes, and
G05 remains `Partial`.

The 2026-07-12 follow-up adds an offline plant-contract v1 compiler without
changing any current trainer. It binds explicit units, the 31-joint order, one
latent physical model, separate engine fit/probe evidence and a calibrated
support envelope. Non-zero `dimensionless <-> N*m` conversion is impossible by
construction; only exact zero crosses that helper boundary. Runtime preparation
also rejects an out-of-support load/speed/temperature/pose request. The final
MuJoCo adapter is required to target the Agibot vendor Gate3/Gate3B runtime and
bind its MJCF, runtime source and 31-joint instantiation report; a generic
MuJoCo wrapper cannot close this gate. Tests exercise the compiler only. No
calibration artifact, runtime wiring or `SC` training arm exists, so G05 stays
`Partial` and current `SZ/SP/LZ/LP` recipes are unchanged. Interface and
commands are in `docs/interfaces/plant_semantics_contract.md` and
`docs/operations/prepare_semantics_correct_plant.md`.

That current execution-contract selection has already produced one early
checkpoint result. Fresh SZ seed1 regressed on q10 from `0.90` at 2000 to
`0.50` at 4000, then consumed one exact K=100 paper. Model 2000 returned
FH/BH/aggregate `33/50,50/50,83/100`; model 4000 returned
`0/50,50/50,50/100`. Model 2000 is retained; the arm continued unmodified at
that paper's decision time and was only later stopped by the separate 2026-07-13 operational
resource decision. Both cells were fresh/exact and had zero physical falls, but all
questions ended through the non-physical post-strike guard; this is isolated
checkpoint selection, not recovery or deployment evidence. Bindings are in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json`.

The fresh/exact Isaac companion reused the byte-identical K=100 schedule and
scored both checkpoints FH/BH/aggregate `0.98/1.00/0.99`, with one guard
reset and no physical falls. It does not reproduce MuJoCo's `0.83` versus
`0.50` ranking. The final earlier-checkpoint tie-break is only a rule for a
complete Isaac tie, not cross-engine support. Model 2000 remains retained
inside the MuJoCo pair; no cross-engine/formal deployment gate closes. See
`configs/phase1_SZ_seed1_2000_vs_4000_q50_isaac_result_20260711.json`.

The original cadence no longer serializes fresh milestones behind causal terminal.
Each Pod now has independent original-causal and original-fresh manifests and
state directories, matching the scale-out split. Current q10 manifests carry
both a fail-closed top-level screen policy and per-job `screen_only=true`; the
checked-in worker rejects omissions/contradictions, verifies `schedule_k`, and
binds a canonical screen-policy-plus-job contract SHA before a completed state
can be reused (while recording the full manifest SHA for audit). This closes
the documented silent parameter-reuse path while preserving the rule that only
a separate q50 paper may stop or promote an arm.

The split itself exposed a Pod-specific historical gap: seed1 `model_4000.pt`
had not existed when the old combined Pod1 worker was replaced, whereas Pod2
seed2 4000 had already been judged. The Pod1 fresh queue therefore starts at
4000 and Pod2 at 6000. Only the childless Pod1 fresh worker was precisely
restarted; no trainer or judge child received a signal.

Before copying or launching any checked-in queue, run
`python3 scripts/validate_phase1_queue_governance.py`. It validates all 142
scale-out jobs and 24 cadence plan slots, requires K20/10-per-side q10
screen-only semantics and milestone/barrier continuity, and rejects q50 from
the generic worker. For one milestone-major runtime manifest, use
`--manifest /absolute/path.json --require-readiness-barrier`.

The corrected Pod2 terminal q10 has now finished: M2-old/S1 aggregate return
is `0.40/0.35`, with FH `0/10` for both and BH `8/10` versus `7/10`. It is a
causal/inexact 20-attempt direction screen, not evidence to kill S1 or select
old; the immutable q50 follow-up remains required. Machine-readable hashes and
the paired delta are in `configs/phase1_M2_terminal_q10_pair_20260711.json`.

### 2026-07-12 reward-composition and PhysicalBall source boundary

Post-strike balance recovery, convergence to a shared ready set and arbitrary-time next-task
readiness occupy one phase and may interact, but the 2026-07-13 primary-source audit does not
classify all three as rewards. G05 does not accept three independently positive reward ablations
as evidence that their sum is optimal. Safety remains a non-compensable constraint; ready shaping
is potential progress to a set; random arrival is first a frozen environment/question/deadline
axis scored by the actual next strike. If T1 still needs shaping, run scale-matched balance/ready
`2^2` with paired seed blocks. A full `2^3` is allowed only after an independent readiness critic
is locked on separate train/calibration splits and passes a one-shot preregistered critic-gate q50
that is disjoint from sealed formal Gate3B q50, without hidden-future leakage. Surviving terms use a
constant-budget mixture plus a second total-budget level. The exact design is recorded in
`docs/research/phase1_ablation_acceleration_2026-07-11.md`; no recovery arm has launched yet.

Isaac PhysicalBall Phase-B now has a contract-bound source implementation at `612f54d` with
attempt-generation tokens, served/contact/landing validity, held publication and strict
training/T1 isolation. Focused host verification passed (`63 passed, 1 artifact-gated skip`).
This is only a source gate: no Pod Isaac runtime result exists, a clean-detached 100-row ledger
has not been produced, and moving-racket substep geometry is not yet quantified. G05 therefore
remains `Partial`; existing analytic Isaac scores cannot be relabelled physical.

The fresh SZ model-2000 four-seed exact MuJoCo q50 is also complete. Seeds 1/2/3/4 return
`83/100`, `100/100`, `100/100`, and `20/100`; seed 4 has FH/BH=`0/50,20/50`, with zero
physical falls. Median `.915` passes, but the preregistered worst-seed, spread and worst-side
criteria fail, so the checkpoint evidence is not seed-stable. At this q50 decision time all trainers
continued unchanged; the result authorized neither stopping nor promotion. The separate 2026-07-13
owner resource decision later stopped seed1/2/4 without rewriting this paper. Later same-milestone curves determine
whether seed 4 is delayed or persistently sensitive.

### Model-4000 four-seed matched paper preregistered (2026-07-12)

The next fresh `SZ` milestone is now queued without touching a Pod. It freezes
seed1/2/3/4 at `model_4000.pt`, the exact model-2000 K100 file bytes
`66e89986...71cb3`, semantic SHA `7dc6af82...ff3e`, question order
`b87e81a3...1f91`, and the same four stability thresholds. It cannot prepare
or start a q50 judge: the queue has no runtime entrypoint, and its validator
contains no SSH, process launch or signal surface. Pod1 must first produce a
content-bound finite/iteration/contract/lineage readiness audit for seed1/3,
Pod2 for seed2/4; only their exact four-seed union can create the mandatory
activation artifact. Any absent/non-finite/wrong-iter/wrong-contract arm keeps
the whole paper runtime-ineligible.

The interpretation is deliberately narrower than “family stable.” Seed4 was
`.20` at 2k; it supports delayed learning at 4k only if it reaches the unchanged
`.65` aggregate and `.50` on both sides. But seed1 4k was already known before
this preregistration to be `.50` aggregate (`FH=.00`, `BH=1.00`). Therefore the
four-seed 4k stability gate is mathematically unable to pass its unchanged
worst-seed threshold, even if seed4 recovers. This matched paper can diagnose
the seed4 trajectory and full distribution; it cannot launder the family into
a stable baseline or change training. Source validation passed `20` tests.
Queue/prereg/validator SHAs are `d4e69d91...d3909`, `ca5ea90f...bff0`, and
`e763ecb9...6cd3`; commands and the future-runner boundary are in
`docs/operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md`. G05
remains `Partial`; no Pod audit, activation, runtime or hardware action has run.

### Recovery tuple is now a structural axis, not a reward bundle (2026-07-12)

A read-only audit of the implemented T1 source and the current vendor Gate3 runner found a concrete
train/deploy mismatch. Natural wrap training installs one complete new
`position/velocity/normal/side` question tuple; T1 keeps the complete previous tuple until reveal
and then installs the complete next tuple atomically. Actor latency/dropout likewise carries all
four fields as one generation. The current 179-D deploy recovery path instead feeds a **new,
live-base-anchored position** together with the **previous strike velocity and previous strike
normal/rho**. No bound training transition creates that mixed-generation moving target. It is now
classified OOD and is not a formal arm; tuning its anchor does not repair the missing semantics.

`configs/phase1_recovery_tuple_abc_prereg_20260712.json` freezes the replacement comparison at SHA
`ca7806df...d810616`:

- A: an interruptible, content-bound safe PD/trajectory bridge into the ready set;
- B: the same actor receives an atomic canonical tuple consisting of a ready-set-selected racket
  position, zero desired velocity, neutral ready normal, rho zero and ready-phase semantics;
- C: the actor retains the complete previous tuple until the atomic next-question reveal.

Ready is a safety-and-reachability **set**, not equality to clip frame 0. It jointly requires station
and upright tolerances, low base/joint/racket speed, stable contacts and slip, joint/torque/q-des
margin, self/table/net/ground clearance, and a bounded safe start to every enabled next motion,
question family and random-arrival deadline. If the global intersection is empty, the design must
declare family-specific ready sets and an explicit transition graph rather than silently deleting
hard questions.

Existing finite, lineage-bound 179-D atomic-question checkpoints have only two narrow uses: A may
reuse one as a frozen **swing diagnostic** after bridge/handoff certification; C may run a zero-shot
coherent-tuple diagnostic. B requires fresh training because current checkpoints never saw the
zero-velocity/neutral-normal canonical tuple. A fair A/B/C causal comparison is fresh exact and
paired; C also needs fresh training before any learned random-arrival recovery claim. No old model
may be relabelled T1-trained.

A also needs a fresh checkpoint for the formal comparison: an external bridge changes who owns the
executed action and therefore changes the PPO data contract. Every tick must bind
`actor_control_mask`, executed action, shadow action, last-action observation and loss masks. Only
an actor-owned, actually executed sample has a valid policy logprob. Bridge ticks have zero policy,
entropy and value loss masks; shadow actions are diagnostic only. The actor's last-action channel
must be the exact content-bound projection of the executed bridge action, never shadow/zero/stale.
Actual bridge rewards use duration-correct `gamma^k`, collapse into the preceding actor option
transition and use `gamma^duration` to bootstrap at the next actor-controlled state unless the
simulator truly terminates; miss/infeasible rows are not fake
terminals. A prebound common budget fixes rollout env-steps, scheduled opportunities, updates,
actor-controlled samples, minibatches and epochs across A/B/C. B/C surplus samples are
deterministically downsampled without seeing outcomes; an A shortfall fails the whole paired update
instead of padding/reusing samples or running extra A steps. Evaluation keeps the full scheduled
denominator regardless of ownership masks.

The first structural paper freezes reward source/weights, total reward budget, motion, bank, face,
plant, network, observation/action schema, seeds, optimizer and random-arrival rows. It prohibits
mid-sequence robot/last-action/history/noise reset and deadline shifts. A's handoff remains blocked
until the exact executed-bridge-action projection into actor history is content-bound; shadow,
zero and stale action substitution are prohibited. Only if B/C fail ready-set acquisition without
single-strike regression may reward work begin. Random arrival remains an immutable environment
axis and actual next-task objective. Normalize balance-absorption debt and ready-set potential on
frozen rollouts, then run paired `2^2` presence/absence. A third readiness potential and full
`2^3` require separate critic train/calibration splits, a one-shot disjoint critic-gate q50 and proof
that no future tuple leaks before reveal; formal Gate3B q50 remains sealed. A
constant-total-budget simplex may follow only for surviving components, and must include a second
total-budget level because fixed total alone identifies proportions, not PPO reward magnitude.
Positive hold income is still prohibited, and safety/self-hit can never be offset by another reward.

The literature boundary is now explicit in
`docs/research/phase1_continuous_rally_timing_2026-07-11.md`: ACE supports an interruptible
near-time-optimal reset bridge and conditioned prepare posture but has no free-standing humanoid
balance debt; HITTER changes tasks only after a swing completes; SMASH's phase/recovery clips feed
offline library generation and runtime motion matching; PACE's five serves are not arbitrary
mid-followthrough reveal. These systems motivate T0/T1/T2, but none proves this A3 policy, vendor
MuJoCo plant or random-arrival contract. G05 therefore remains `Partial`.

2026-07-13 的新 reward 次序目前仍是文档级设计：现有
`phase1_recovery_tuple_abc_prereg_20260712.json` 与 validator 继续强制旧的三 reward/full `2^3`。
它们未被追写，也不得冒充新设计的 E1 证据；必须生成新的内容寻址 prereg、validator 和测试后，
才允许按“先 `2^2`、校准后可选 `2^3`”点火。这个同步缺口使 G05 继续为 `Partial`。

The pure-contract validator passes `50` red-team tests, including nested duplicate-key, non-finite
JSON, strict type identity, exact identity/time/scope and unknown-key rejection; `launch-check`
deliberately fails because
the schedules, checkpoint inventory, bridge and trajectory certificate, canonical tuple selector,
fresh checkpoints, A ownership/PPO contracts, full numeric ready contract, Isaac continuous judge,
vendor Gate3 runtime/stability judge, Gate3B behavior judge, shared Gate3/Gate3B runtime, self-hit
instrument and calibrated plant are not bound. Gate3 is the exact-C++/MJCF/plant/model first-tick and
continuous-stability hard prerequisite; Gate3B reuses that runtime with the immutable random-arrival
q50 and is the final first-strike/return-quality behavior arbiter. Commands are in
`docs/operations/run_phase1_recovery_tuple_prereg.md`. No simulator, Pod, GPU, C++, Gate3 worktree
or robot was changed. G05 remains `Partial`.

### 2026-07-12 formal train-normal envelope export

The 179-D export chain now consumes the demanded-normal rows of the exact schema-3 train bank
instead of binding only its file SHA. Native Isaac export requires the live validated
`QuestionBank`; the Isaac-free standalone path requires `--train-bank`, runs the same strict bank
and motion-anchor loaders, and refuses to inherit any `stage1_*` envelope from a donor ONNX. For
each clip independently (clip0 forehand, clip1 backhand), rows must be unit within `2e-4` and stay
strictly more than `1e-6` inside the raw +Y/A-frame reference hemisphere, exactly matching the C++
runtime gate. A merely positive but `<=1e-6` row now fails export. The checkpoint contract and both
exporters must also carry the exact sign table `[+1,-1]`; representability is
`sign[clip] * raw_A.x > 1e-6`, not `raw_A.x > 0`. Therefore forehand raw A is positive-x and
backhand raw A is negative-x, while the external schema-2 physical striking-face B remains
opponent-facing positive-x for both. The normalized per-clip raw-A row sum defines the
spherical-cap center and the minimum row dot defines its boundary. The exporter records the
frame, convention, algorithm, tolerances, exact sign table, centers, references, thresholds, row
counts, train-bank SHA and source-family SHA in a canonical payload with its own SHA-256.

The external B normal is converted to A only after clip selection; position and velocity are not
changed. Dependency-light Python verification covers all bank rows at the exported boundary,
independent forehand/backhand signs, a physical-B positive-X but raw-A out-of-support normal,
opposite-sign poisoning, non-unit rows, wrong clip order, bank/family mismatch and payload content
binding. A subprocess smoke imports the standalone exporter without the package `__init__`, Isaac,
ONNX or Torch. The prospective real-bank fixture
`configs/phase1_face179_real_bank_envelope_expectations_20260712.json` binds bank
`2da2bd12...a0700`, family `b21c161a...28ad5`, `757/724` rows and expected cap minima
`0.974278/0.972078`; those read-only statistics are export expectations, not behavior evidence.
The standalone path now validates checkpoint binding, donor, both motion files, harvested buffers,
bank and derived envelope before creating any graph output. It exports to an owned same-directory
temporary file, checks the graph and metadata round trip, fsyncs it, then atomically replaces
`policy.onnx`; validation/export failure preserves an existing final model and removes the temp.
Behavioral tests cover successful replacement, injected failure and empty output. Host verification
is `41 passed, 1 optional real-runner integration skip` for the focused contract/export/preflight
group and `11 passed` for the planner wire. Pod1 subsequently produced an envelope-bearing formal
SZ seed3 model-2000 ONNX (`0c428ddf...b7b155`) from the exact train bank and passed the full
ROS/AimRT Release suite plus strict positive/negative production preflight; see
`configs/gate3_face179_strict_preflight_evidence_20260712.json`. This closes an export/model-load
prerequisite only. No policy has yet passed vendor MuJoCo backend first tick or behavior with the
envelope, self-hit gate or recovery contract. G05 remains `Partial`.
### Yikang branch changes were integrated by current-main semantics (2026-07-12)

Three small changes were audited against `origin/main@b2067ba`; neither old-base branch was merged
whole. The fit-lineage NumPy oracle from `stage1-fixed-point@bc86995/f0ac2fb` was accepted and
hardened. The existing Torch parity test now defaults to that in-repo reference, emits the exact
reference/contact-model/venue-YAML SHA tuple, and fails on an explicitly missing `RECORD_DIR`.
Normal handling is scale-invariant and rejects zero/NaN/Inf instead of propagating NaNs. Seven
dependency-light oracle tests pass. The full current-source Torch CPU parity gate also reports
`ALL PASSED`: table/paddle contact errors are below `4.63e-9`, flight RK4 is exact at reported
precision, and first-landing error is `0.000 mm`.

The `head_discipline` diagnosis from `407a443` is retained as a candidate, but its code and `-0.5`
weight were deliberately not ported. That commit is based on old `hitter@5c346ea` and enables a
`HOPEPingPongHitterPureRallyFinalV2` recipe that does not exist on current main; FinalV2Plus also does
not exist here. Importing the term/whitelist alone would create a stale, unnamed reward surface.
Moreover, `origin/hitter@0fccc3c` has a passive-head FinalV3 action contract for the same symptom,
and FinalV2Plus derives an exact reward-key set from FinalV2. A reward term must therefore be an
explicit named-recipe decision, not accidental inheritance or silent stacking with passive-head.

Two dependency-light guards verify that current main has neither the old FinalV2/FinalV2Plus recipe
nor a silently exposed/enabled `head_discipline_weight`. No training was launched. If reward-side
head discipline is adopted later, start from an explicit `0.0` current-line interface, compare it
paired against unchanged control and passive-head, and treat its overlap with balance/recovery/
ready-state rewards as a mixture interaction under fixed total budget. The old `-0.5` is a hypothesis,
not a validated default. In the local Torch/Hydra environment, the current reward/config tests pass
`88` relevant cases. The unfiltered suite has two additional failures in pre-existing
`MotionLoader` handling of a single `Path` as an iterable; neither touches this diff or head rewards.
Because no head code/config was changed, current reward bytes and behavior remain unchanged. Full
audit and commands are in `docs/research/yikang_selective_integration_20260712.md`. G05 stays
`Partial`.

### 2026-07-12 inexact first-actor-candidate observability boundary

The deploy runner's new `--first-tick-json` diagnostic does not change training, the 179-D actor,
actions, normalization, reward or any checkpoint. It records the first observed planner-engaged
actor candidate; idle/wait/recovery rows cannot consume the capture. This is not an atomic planner
snapshot or Gate3 certificate.

Backend `RobotState` has joint q/dq and IMU state but no root linear velocity. The diagnostic reads
the vendor pelvis-twist topic through a subscription-only sim sidecar instead of fabricating zero,
and records observation base separately from the joined vendor-world base. Missing/stale/nonfinite,
regressed headers and odd generations fail closed. ONNX Runtime loads the same stable bytes whose
digest enters the JSON.

The vendor topics have asynchronous publish-time stamps and no common MuJoCo sample sequence; the
current planner also lacks merged same-tick snapshot/shared payload epoch semantics. Therefore the
outer document and payload fix `evaluation_contract_exact=false`, with planner/native/source/runtime
exactness fields all false. Dependency-light source tests pass `6` cases. No actor/model weights
were evaluated and no Isaac or MuJoCo rollout ran. G05 remains `Partial`; this instrumentation
cannot promote a checkpoint or repair the four-seed stability failure.

### 2026-07-12 model-4000 activation consumer source gate

The previously runtime-free fresh `SZ` model-4000 queue now has a separate reviewed execution
contract and activation-consuming runner without changing the frozen queue, preregistration or
validator bytes. Every command requires the exact all-four activation file and caller-supplied SHA,
then revalidates both content-addressed Pod audits and all four finite/iteration-4000/schema-3/
exact-lineage/hard-contract records. A Pod also rehashes and re-audits its own two checkpoint files
and adjacent contracts before it may prepare or run.

Preparation is no-clobber and only copies the already-materialized K100 bytes; there is no schedule
generation path. Pod1 is fixed to seed1 then seed3 and Pod2 to seed2 then seed4, serially. Seed1 is
conservatively rerun on the same paper rather than reusing its previous score. Each judge uses the
pinned `judge.sh`, the shared `/workspace/.kit_boot.lock`, a new session with observed PID=PGID and
preserved state/log on failure. The runner has no SSH or signal API and cannot stop a trainer or
worker. Result validation binds exactness, 50 attempts per side, question order, MJCF,
execution/ready-state, checkpoint/contract and report/summary/attempt-ledger SHAs before a Pod
result can exist.

The aggregate cannot return a family-stable PASS: known-before-prereg seed1 model-4000 was `.50`,
below the unchanged `.65` worst-seed rule. It only classifies seed4 as delayed learning when 4k is
at least `.65` aggregate and `.50` on both sides; otherwise weakness is persistent through 4k.
Either outcome keeps all training unchanged and authorizes no promotion/deployment/hardware.
Queue plus runner focused source tests pass `40` cases. At source merge no Pod readiness audit,
activation, preparation or judge had run, so that result was not training evidence.

The barrier was subsequently materialized outside both frozen checkouts on 2026-07-13 local time.
Pod1/Pod2 audit file SHAs are `3fc325e1...247b8` and `4f25786b...565f7`; their exact union produced
activation file SHA `9dea76c2...ce704`, content SHA `eaa92ca2...aa4fb`. The immutable source,
schedule, both audits and activation now occupy identical absolute paths on both Pods. Both
activation-consuming `contract-check` calls passed, and the immediate snapshots contained no
child judge, Kit process or Kit-lock holder. No `prepare`, rollout, score, trainer mutation or
signal occurred. This closes only the all-four readiness barrier; G05 remains `Partial`.

Both Pod runtime contracts were subsequently created by the no-clobber `prepare` command. Pod1's
file/content SHAs are `2b76a5a...8201e` / `36e878f0...5ba73`; Pod2's are
`dbecc102...d1c9b` / `91a0070a...30794`. Direct runtime-binding validation rehashed the local two
checkpoints per Pod and confirmed iteration 4000, finite tensors, exact lineage and the common
hard-contract SHA; both train/eval checkouts remained exact and clean. Each contract still records
`prepared_not_started`, `jobs_started=0`, `auto_start=false`, and the post-prepare process/lock
snapshot was empty. No judge, score, rollout, signal or hardware action ran. This is execution
paper preparation, not training evidence; G05 remains `Partial`.

### 2026-07-12 training-backend boundary

G05 continues to own the Isaac first loop and its checkpoint lineage, but a higher Isaac training
score or another Isaac-only reward/teacher sweep no longer counts as resolving G06. Native MuJoCo
training/fine-tuning is now a P0 implementation and promotion track recorded in G06. Shared
observation, action, reward, reset and export contracts must still be updated here when they change,
and the exact fresh `SZ model_2000` checkpoint is the first handoff candidate. No backend code or
training run exists yet, so G05 remains `Partial`.

### 2026-07-12 MuJoCo-v0 handoff and warm-start correction

The native-MuJoCo preflight keeps the 179-D Isaac actor as a source candidate but corrects the
handoff semantics. `load_actor_tolerant()` is not suitable for the planned causal paper: its strict
path restores the runner, while its fallback loads every shape-matching state tensor, including any
matching critic tensor. The v0 loader must construct a seeded fresh critic/optimizer first, then
load only the complete actor, action-distribution state and actor normalizer; tests must prove the
critic is unchanged, optimizer state is empty and iteration is zero. The actor normalizer remains
frozen for the initial v0 paper.

The current schema-3 hard contract also deliberately omits reward weights, termination thresholds
and optimizer settings. That remains valid for its existing curriculum purpose but is insufficient
to identify a MuJoCo causal fine-tune. The new backend experiment contract must additionally bind
those fields, reset/timeout semantics, effective MuJoCo profile, runtime action post-processing and
source-checkpoint SHA. Its first one-shot `Trainer-v0` optimizes balance/strike-state only; the vendor MJCF has
no physical ball/table/net, so it cannot book formal return evidence. Full reasoning and read-only
commands are in [the MuJoCo training-v0 preflight](../research/mujoco_training_v0_preflight_2026-07-12.md).
No code, config or training changed; G05 remains `Partial`.

### 2026-07-13 MuJoCo preflight 红队暂缓合入

franco 已明确批准 native MuJoCo feasibility/implementation 作为 P0 能力轨，但它不是几天内
`Gate3-D0` vendor planner+policy 演示的前置。当前 matched paper 证明解析回球/击球执行跨引擎
塌陷，而 physical-fall 计数接近零，不能夸成“平衡也已证明退化”。候选 `6e5fce3` 的七个授权位
保持 false，focused 63 项与顶层 `468 passed, 9 skipped` 通过；但 action tape/trace 未覆盖
clamp/runtime adapter、静态 source closure 有 alias/exec 逃逸、JSON 接受 duplicate/nonfinite、
MJCF `compiler strippath` 解析错误。因此当前 `NO-MERGE`，四项必须先补负测并修正。

第一个 single-env core 还必须对 N=1/8/32/64 分别报告 sim-only 和完整 rollout+一次 PPO update
的 step/s、RTF、RSS/CPU 与扩展效率，并按预注册 transition budget 推算两臂×两 seed 墙钟。
只有能在 48 小时内完成且留 30% 余量，才继续 CPU-Python 长训；否则转 C++/OpenMP 或另行过
parity 门的 MJX/MJWarp。没有 `VecEnv`、PPO、训练、Pod、simulator 或真机行为证据，G05 仍为
`Partial`。

### 2026-07-13 v12、高点拍压、横移老师与非击球臂设计

`configs/motion_video_intake_20260713.json` 已逐字节绑定 7 段新的私有视频：v12 正反手挡球、
一个反手高点拍压第五动作，以及左右横移各两个下肢老师候选。7/7 文件核验与 11 项专项测试通过。
其中 Franco 主线的 S0 高点拍压和 M0 四条横移又在 Pod1 完成 exact GVHMR：帧数分别为
`88/88` 与 `105/105、97/97、82/82、96/96`，所需 tensor 全 finite，输入、输出、queue、binding
和 audit SHA 已进入 `configs/motion_video_gvhmr_s0_m0_results_20260713.json`。这仍只是人体结构证据；
五条都没有完成 GMR、运行顺序 schema-2、L0、厂商 L1、桌网余隙、动力学或匹配题目的回球门，
也没有候选进入 RL 队列。v12 本轮未执行。

横移素材被定义为“以横移距离为条件的下肢老师”，不是另一种挥拍。上下半身先按准备/击球支撑/
恢复事件对齐，再明确根节点、骨盆、躯干和足接触的所有权，由受约束的全身求解处理耦合；TOPP
只能给已接受路径重定时，不能把错误足接触变稳定。恢复终态要回到该素材初始的水平双脚相对向量，
同时保留站距和前后错位。v12 必须在挡球专用考卷上赢过旧安全候选，高点拍压必须先过独立高球卷，
之后才允许讨论四动作对五动作。

另一个配对设计测试是否解除左侧非击球臂的模仿，让它参与平衡；“直接移除 Reward”和“固定总预算
重新分配”必须分开，所有硬安全保持开启。三份记录见 [实验登记册](../experiments/README.md)。
后续 E1 source gate 已把 A0/A1 直接 mask 物化：四条 body-imitation Reward 都显式列出
`body_names`，A1 只删左 shoulder/elbow/wrist，并保持 A0 的躯干/右击球臂、所有权重和硬安全不变；
contract drift 与错误布尔值 fail closed。machine prereg 固定 fresh seed17、`4096 env × 1001 update`、
`+200/+500/+1000`，默认 plan-only，Pod launch 需要 root 显式 token，claim/checkpoint/result 都
no-clobber。checkpoint 内嵌 hard contract 还逐臂绑定 post-override 四项 body list；两臂 hard SHA
必须不同，而删除该唯一字段后合同必须完全相同。源码/runner 共 `71 passed`，但 Pod
runtime 已形成一个受控 partial：A0 于 `2026-07-13T19:48:35Z` 以 PID=PGID `1811464` 启动，
`19:49:15Z` ready；其 `model_200.pt` 已验证 embedded iter `200`、finite、fresh lineage `1` 并绑定
hard-contract SHA `14ef410b...29f1`。旧 outer verifier 随后因把 schema-3 bank metadata 的 physics SHA
错当 compact hard-record direct leaf 而精确假拒绝，故 A1 从未 claim。v1r1 source gate 现改为独立解析
bank file/metadata、复现旧错误、先 attest 既有 A0 三份稳定 SHA，再且仅再 claim A1；A0 dead/drift、
A1 预存在或 bank drift 都 fail closed，且禁止重跑 A0。v1r1 专项 `12 passed`，新旧 runner 合跑
`30 passed`。现场 `validate-runtime` 全绿后，唯一一次 `launch-a1` 已成功：A1 PID=PGID `1816234`、
Kit ready，emitted hard-contract SHA 为
`c85b52a28ad64a667a7b522562842466270b3741591f6daf09afc1d0f7c6b146`；A0 PID=PGID `1811464`
untouched。recovery/runtime receipt 已 no-clobber 写入，judge 未启动。external `--mode plan` 另暴露只读
相对路径 bug：它在 external control 下误找 `control/configs` 并在任何写/claim 前失败；runtime/launch
使用冻结绝对 v1 路径，不经过该分支。已绑定的 v1r1 bytes 不得修改，路径 bug 只能在新版本修。
A1 milestone、配对终档和同卷判读仍未发生；A2 固定预算继续 blocked。详见[实验](../experiments/non_striking_arm_imitation_ablation_20260713.md)
与[操作](../operations/run_phase1_non_striking_arm_imitation_a01.md)。S0/M0 有 Pod 离线结构结果，但没有
Isaac/MuJoCo 训练、仿真行为或真机动作，G05 仍为 `Partial`。

S0/M0 的 post-GVHMR machine handoff 已完成 exact runtime `consume`，输出分别是 4,970/9,242 bytes，
SHA-256 `d57a93e0...a1054` / `60c55150...088ef`。下一层 canonical-beta 已拆成两份独立 no-clobber
计划；consumer 只注入旧 exact donor，其他 PT leaf 必须 save/reload bit-exact。host 新旧专项为
`15 passed, 1 skipped`；真实 canonical-beta `inspect/consume` 也已在绑定 Pod1 CPU runtime 完成，S0/M0
completion manifest SHA 为 `964a7333...f1be3` / `5cef05f7...71a65`，五条 non-beta 内容 bit-exact。
这只解锁另建 exact GMR prereg，不直接解锁 schema-2 或 RL。S0 仍不得借用拉球题或声称击球有效；M0 的
canonical foot-site 与容差现已由 exact-GMR prereg 冻结，初末二维脚间向量和 pass 仍为 null，必须由
robot-coordinate GMR 产生，双脚
并拢不能通过。详见 [handoff 记录](../experiments/motion_post_gvhmr_s0_m0_handoff_20260713.md)与
[canonical-beta 记录](../experiments/motion_canonical_beta_s0_m0_20260713.md)。

### 2026-07-14 S0/M0 exact GMR prereg boundary

The five canonical-beta PTs now have separate S0/M0 exact-GMR machine plans, but they do not yet
authorize a training asset. The CPU consumer freezes the old GMR argv, exact source/runtime/model
closure and report-last publication. M0 additionally freezes exact 30 Hz ready-window sample lists,
canonical A3 foot FK and both components of `right_foot_xy-left_foot_xy`; a finish narrowed by more
than 5 mm fails independently of the 3 cm component band. S0 contact/effect remains null and cannot
borrow a loop paper.

The 2026-07-14 follow-up bound all 16 exact source/runtime facts. Direct retarget XML has the bound
31-joint/32-body order but an exactly empty site inventory; canonical vendor foot sites cannot be
substituted into retarget evidence and remain exclusive to M0 stance FK. Both batch plans and shared
runtime are `preregistered_not_executed`; both host static validations pass. Runtime
`inspect/consume` and GMR outputs still do not exist. Schema-2, L0/L1, dynamics and RL remain blocked;
no trainer or hardware command ran. G05 remains `Partial`; details are in
[the experiment](../experiments/motion_exact_gmr_s0_m0_20260713.md).

### 2026-07-12 文档路由与当前成绩表

训练状态现在按职责拆分，不再复制到三份流水中：

- [`docs/NOW.md`](../NOW.md) 负责当前完整训练流程、现行课程阶段、各主题的问题/解法/效果/差距，
  以及最接近正式目标的逐动作单拍/连续成绩表；
- [`docs/experiments/`](../experiments/README.md) 负责假设、冻结变量、run 表和决定；
- [`docs/TIMELINE.md`](../TIMELINE.md) 只解释已经进入 `main` 的重要逻辑变化。

全局优先级只看 [`docs/NOW.md`](../NOW.md) 的统一工作队列；GPU/Pod 不再各建影子队列。
排序、Kit boot lock、关键路径独占与广度波 3–4 条/卡的适用条件统一见
[跑批作战手册](../runbook.md#统一队列排序与算力纪律) 和
[`run_on_runpod.md`](../operations/run_on_runpod.md)。本次只收口文档规则，没有改发射命令。

当前 `SZ model_2000` 成绩表汇总四个 exact seed，没有隐藏 seed 4：正手单拍的
无物理摔倒/解析击球/解析回球为 `200/200, 137/200, 133/200`；反手为
`200/200, 170/200, 170/200`。连续格仍为 `未测`，因为每道 K100 题都通过非物理
tracking guard/reset 路径结束，其中包括 seed-4 的击球前 guard。这是同一份既有结果，不是新实验；
这里的击球/回球来自 `VirtualReturnScorer` 对拍状态的推演，不是 simulator 球-拍-台接触。
seed-stability 判决仍为失败。本次文档迁移没有运行 Pod、模拟器、训练或真机动作；G05 仍为
`Partial`。

### 2026-07-13 formal tuple source integration boundary

The accepted deployment repair changes planner transport and actor engage safety only: no reward,
training recipe, observation/action dimension, checkpoint or active Pod arm changed. Formal racket
schema 3 now references one exact formal base sequence, while closed-loop gates and the actor
observation use latest tick-start base. Latest stale/low/implausible/epoch-changed or
revocation-changed base blocks actor inference during active swing and recovery; ordinary valid
same-epoch refresh remains legal.

Exact source passed planner tests `180 passed, 2 optional skipped`, Pod2 portable Release focused
`40/40` and native `233 passed, 5 optional skips, 0 failed`. These are source/binary results, not a
new training setting or behavior score. The 179 actor still needs ROS/AimRT first tick and vendor
MuJoCo behavior; G05 remains `Partial`.

### 2026-07-13 persistent q50 top-level startup source gate

The model-4000 activation consumer now has a separate
[persistent-supervisor contract](../interfaces/q50_persistent_supervisor_contract.md) for the one
remaining process-lifetime gap: the consumer's top-level Python process could disappear with its
invoking SSH shell while an already-detached judge child continued. The new wrapper exposes only a
manual no-clobber `launch` and read-only `inspect`. It neither changes nor replaces the existing
consumer, execution config, all-four activation, prepared runtime contracts, [q50/K100 paper](../DEFINITIONS.md#q50-and-k100),
checkpoints, trainers, or workers.

Before execution, the child creates a new session, redirects fixed stdio, closes inherited file
descriptors and publishes a hello with `PID=PGID`, Linux boot id/procfs start ticks, bound executable
SHA, exact argv/fixed-environment digest and the complete source/config/activation/runtime SHA
closure. The parent publishes an immutable ledger and commit token only after independently
validating that identity. Without the token the child times out and exits by itself; after the token
it revalidates all bytes and identity/token/ledger/result before acknowledgment and again before
`execve`. First possible visibility of the token's final link, not acknowledgment timing or the
following directory fsync, is the irreversible no-retry point; the startup deadline only governs
token absence. Independent acknowledgment and exec observation
windows return `token_published_pending_ack` or `committed_pending_exec` with return code zero when
progress is not yet visible, while every second launch remains no-clobber rejected. Inspection
rejects PID reuse and delegates terminal acceptance to the original runner's full result validator.

The focused supervisor suite passes `24` cases; queue+consumer+supervisor together pass `64`. The
suite includes tokenless deadline expiry, post-token delayed rehash, a 1.15-second acknowledgment
atomic-publication stall, post-ack delayed exec and terminal-result A-to-B replacement. The three
post-token stalls preserve no-retry authority and converge without a fatal-before-later-runner
sequence. Post-link token-directory-fsync plus evidence-stat failure, token temporary-cleanup
failure and parent-observation-write failure also return committed pending, reject restart and later
inspect as exact running; none can escape as a retryable launch error. The host is macOS, so procfs behavior is covered through an injected
identity seam and still needs one Linux fake-runner source smoke before any real q50 process. No Pod
deployment, judge, simulator, training mutation, process-control action or hardware command ran.
G05 remains `Partial`.

### 2026-07-13 Phase-1 运行池运营裁剪

负责人明确批准把已显示持续塌陷的 fresh 运行停止，以便把算力换给
[NOW 唯一队列](../NOW.md#统一工作队列唯一优先级账本)中更靠前且前置已满足的工作。16 条
fresh 广度臂中首先精确停止 8 条：formal `SZ` seed1/2/4，以及诊断格 `SP` seed1/4、`LZ` seed1、
`LP` seed1/2；其余 8 条当时继续，后来按本节下方的 signed-face 取证再停止。详细 q10 曲线、
已知 q50、PGID、最后 checkpoint 与 SHA 见
[拍面×plant 广度实验](../experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)。

这是负责人在结果出现后作出的**算力运营决定**，不是发射前预注册的统计停止规则。历史 manifest
仍保持 q10 K20/每侧 10 题 `screen_only=true`，不能晋级；model-2000/model-4000 q50 合同中的
`whole_arm_stop_allowed=false` 也不改写。因此不得把这次停止写成“q10 正式 reject 三个 formal
seed”，不得隐藏已停止 seed，也不得用它给任何 setting 晋级。

每臂信号前均保留最新日志/checkpoint，并验证文件名迭代号等于内嵌迭代号、`1,762,715` 个浮点
元素且 `nonfinite=0`、schema-3、fresh lineage=1，以及 checkpoint ↔ 相邻 hard-contract SHA
一致。TERM 未使这些 trainer 退出；确认没有 live child 或 Kit-lock holder 后，只向各臂 `.launch`
登记的 PGID 发送 KILL，再核对该组消失和其余接受臂仍存活。没有使用 broad `pkill/killall`，没有
向 worker/judge 或真机发信号。formal 四 seed 的 model-4000 checkpoint 早已内容绑定并通过
readiness，所以已准备 K100 后续卷输入不变。这个运行处置不新增质量成绩，也不关闭训练稳定性、
signed-face 或连续能力门；G05 保持 `Partial`。

model-4000 与剩余臂的 signed 切面随后使这个运营决定扩展到全部 16 臂：剩余臂最近
24/24 K20 格的正手 signed composite 均为 0，无论 shared 还是 legacy face 都不分离。第二波
也在 no-clobber checkpoint/log/contract 审计后只按精确 PGID 停止；两 Pod 无 trainer、GPU 已空。
这仍是负责人事后算力决定，不是 q10 阈值的预注册 stop rule。详见
[广度实验](../experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)。

### 2026-07-13 model-4000 四 seed matched q50 稳定性失败

已准备的 fresh `SZ model_4000` 同卷在 Linux fake-runner 冒烟后，经过内容绑定的一次性
supervisor 在两 Pod 完成 seed1→3 与 seed2→4 串行判卷。两份 Pod result 均为
`terminal_result_validated`，最终无残留 supervisor/child judge/q50 Kit-lock holder。正式
aggregate file/content SHA 为 `1ba88e39...d195` / `226e6050...648d`，独立 canonical 复算通过。

四 seed 旧 parsed return 为 `.50/.88/.98/.00`，median `.69 < .75`，worst `.00 < .65`，
spread `.98 > .20`，minimum-side `.00 < .50`，四项冻结门全失败。seed4 为
`0/100`且有 21 次 `fall_root_z`，因此归类为“持续弱到 4k”而非晚熟。该失衡是
seed4 特定结果，其他三 seed 物理 root fall 为 0。

同一结果又证明旧解析分不能作 baseline selector：seed2/3 正手 raw-A signed normal 误差
`172.33°/174.35°`，signed strike composite 都是 `0/50`，但 parsed return 仍为
`38/50` 与 `48/50`。所以不会晋级最佳 seed，也不再用相同 `SZ` 续训买晋级证据；
下一步是 `n/-n` 负控、signed-face scorer 修正和同卷复判。这仍是每题重置的
Python BankExam，不是 physical ball、连续恢复或厂商 Gate3/Gate3B；G05 保持 `Partial`。
详细证据见 [Fresh SZ 稳定性实验](../experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。

### 2026-07-13 signed-face 训练信用门：源码闭合，行为仍待 canary

seed3 的 content-bound TensorBoard 摘要把“旧解析尺有病”推进到训练信用本身：正手法向误差从
iteration 2000 的 `167.49°` 继续到 13800 的 `174.02°`，signed normal pass 一直为 `0`，但训练内
virtual return 同期从 `.692` 升到 `.965`；反手在 13800 为 `5.86°/.996/.967`。同一 step 的三项
全局 virtual reward tag 合计 `.4615195`，是全局 normal reward tag `.15587743` 的
`2.960784637×`；但这些 tag 汇总所有环境和正反手，不能归因或量化正手错面的 reward 份额。实际
`env.yaml` 已绑定 `virtual_ball/vb_metrics_only=true` 与 `20/30/5/5` 四项权重。结合 face-blind
源码路径，只支持“wrong-face FH states were treated as reward-eligible by the active face-blind reward
path”，不支持“正手错面实际领取了多少”或把反面行为归因于单一 reward；完整 tag、step/value、
event/training-contract/env.yaml/launch/nohope.diff SHA 与 source-commit claim 证据边界见
[拍面符号卷宗](../experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。

当前 feature source 已把 `hope_commands._vb_evaluate` 的 `vb_fired` 改为：先用统一
`face_tracking_pair` 比较有符号拍面，再要求 achieved/target physical-B 都严格朝 `+X`，最后才允许
`orient_normal` 进入冲量计算。所有 `virtual_pass_net/landing/spin` 都消费这个门后的 one-shot
mask。Torch oracle 同步增加 finite/non-degenerate、strict hemisphere 和 physical-B 门；NumPy
`n/-n` 负控证明出球/落点仍相同而错面不能触发记分。

focused 回归为 `38 passed, 1 skipped`，顶层 broad 为 `546 passed, 9 skipped`；另一个排除
Torch/Hydra import-bound 文件的 training dependency-light 组合为 `381 passed, 21 skipped`。
这些不是 Isaac 行为结果。本节没有启动 Isaac、Pod、trainer、judge 或真机。下一个 fresh 双侧
canary 必须绑定新 source/hard-contract SHA，
验证错面样本不再得到 `vb_fired`/virtual reward，并观察正手 signed normal 与 return 是否共同学习；
它按[单-seed 机制漏斗](../experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)先买机制证据，
不先复制四个 seed。在此之前不能写“训练行为已修复”，G05 继续 `Partial`。

该漏斗的 machine prereg 与 launcher 后续已物化：训练 source 固定 clean detached
`882fea4285f0cf9a97ba79d79ae8af31d26ea1ed`，一张卡只允许 seed3 的 A/B/C/D 四个不同因果格；L1
为 `512 env × 25 update` launch-integrity smoke。热启动 A/B 因旧父合同缺少当前
event-timing/target-cadence 字段，固定为显式 inexact transfer 并要求 lineage `0`；fresh C/D 必须
lineage `1`，四格 emitted contract SHA 一致。四格终档 finite/contract/lineage 全通过后只写
no-clobber completion 证据；它不能单独授权 L2。L2 的 `4096 env × 1001 update` 设计因 immutable
signed directional checkpoint paper 的 path/SHA 未冻结而在 runtime preflight 顶部 fail closed。
focused 静态/攻击回归 `23 passed`。首次 Pod `control/v1` preflight 在任何 run claim 前暴露审计器
只看 checkpoint 顶层的假拒绝；只读递归复核证明父模型 `74` 个浮点 tensor、`1,762,715` 个元素均
finite，合同三元组实际位于 runner 的 `infos`。`control/v1` 保留未覆盖，v2 改为递归 finite 扫描并
强制从 `infos` 绑定 schema/SHA/lineage。v2 随后在 A 格首次 learning iteration 前暴露 source-first
环境未传给 child；失败 claim/log 保留，B/C/D 未创建。v3 绑定 tracked setup 脚本、拒绝 local override，
v3 因在 `SimulationApp` 前真正 import IsaacLab 而假拒绝；v4 改用 `find_spec` 只验证模块 origin 位于
exact `882fea4` worktree，正式 import 留给 Kit boot。当前仍没有新 Isaac checkpoint 行为结果，
不能把 E1 写成训练修复。v4 随后进入 Kit/scene 后发现 detached worktree 缺 Git-ignored A3
URDF/mesh/config；失败 A claim 保留。v5 从 clean `6d93bcb` 恢复并同时绑定 source/target tree 的
`46` files、`15,378,264` bytes 与 canonical SHA `0137f59b...26c6`，claim 前拒绝缺失/额外/symlink。
复现命令、SSH 中断恢复和半写 claim 的 fail-closed 处置见
[操作文档](../operations/run_phase1_signed_face_rescue_funnel.md)。

v5 随后在 scene 构建完成后被 schema-3 loader 正确拒绝：旧 train bank 的 physics contract 记录旧
`virtual_ball.py`，而 `882fea4` 新增 signed-face helper。失败发生在 hard-contract/first iteration/
checkpoint 前，A claim/log 保留，B/C/D 未创建；不能以 legacy load 绕过。main 现加入严格 no-clobber
bank rebind consumer：先证明七个 physics 文件只有一个 helper 定义新增、移除它后 executable AST
相同，且 generator/loader 不变；再要求全部非-meta 数组 raw bytes 不变、metadata 只有四个 leaf、目标
runtime 的 exact motion contract 和 1481 题 old/new contact/flight bytes 相同。Pod1 已正式发布并复核
bank SHA `3a9d8851...5b71` 与 report SHA `9fffed03...bb37`：24 数组未变，正/反手 `757/724` 题的
旧/新输出 raw bytes 相同，landing/net 全过。v6 又把 report closure 及唯一允许的父旧 bank→当前新
bank common-field transition 写入 preflight；其他父/新共同字段仍逐值相同。

实际 epoch-1 v6 后续在 clean `50c49e5` source 上启动：A/B/C 到终档，checkpoint 迭代分别为
`13824/13824/24`，lineage `0/0/1`，共同 hard-contract SHA `dfc583d4...888a5`；D 在
`runtime_verified`/checkpoint 前 Kit boot timeout。其旧 launch/state/log 与 timeout 诊断已按 exact SHA
冻结，PID 已死且旧 claim 不覆盖。原始 checkpoint audit `62076758...d354` 绑定 A/B/C 的 exact
checkpoint/finite/lineage，并明确 D `run_dirs=[]`。后续 [v6r1](../DEFINITIONS.md) 首次真实
`validate` 在任何 claim/训练前暴露 validator 自相矛盾：它错误要求旧 would-be D training path 存在，
而冻结 audit 与 filesystem 都证明该 path 应 absent。团队没有伪造目录；v6r1 从未 claim、launch、
signal 或训练。新 [v6r2](../DEFINITIONS.md) 只做 source contract correction：旧 path 必须 absent，
任何 entry kind 都 fail closed；只支持 `static-validate`，没有 runtime preflight、命令重建、进程检查、
launch、signal 或 finalizer，且明确 NOT LAUNCHED。
后续 foreign v8 使用 clean `72418fff` 与全新 manifest/launcher，`v6_artifacts_adopted=false`，按 terminal
barrier 串行发射 A/B/C/D。A/B/C 前序已终档；D 是第四格，900 秒内再次未出现 hard-contract marker、
runtime verified、learning iteration 或 checkpoint。locked wrapper 只对 `PID=PGID=1782834` 做精确
cleanup 并返回 124；日志无 NaN/Inf/Traceback/OOM/malloc/Killed。因为这已是继 v6 D 后第二次独立
pre-contract timeout，自动 retry 已停止，转入 boot 根因。没有四格 activation；L2/judge/第二 seed
仍固定 false，所以 G05 仍为 `Partial`。操作仍见上面的 signed-face 漏斗运行手册。

后续三次只读审计把 v6/v8 D 的共同失败点收窄到 identical table USD 的 load→PhysX 交界；两份 D
normalized argv 除 versioned run name 和 v8 launch-claim provenance 外相同，且都从未出现 hard contract
或 learning。相邻 C 分别在 `2.339/3.031 s` 越过该边界，v8 D 又是在 C clean shutdown 后 `44 s`
启动，所以这仍不是 reward/seed 学习结论，也不能用配方相同 retry 获得新信息。事后容量非饱和只
降低持续资源耗尽的可能性；Carbonite residue、瞬时 driver/filesystem stall 与 ordinal-4 累积状态仍未
分离。机器账见 [boot 结果 ledger](../../configs/phase1_signed_face_boot_root_cause_results_20260714.json)。
下一份 `D-first × ordinal-4`、`host IPC × private IPC` 的 scene-only 诊断只有
[design-only prereg](../../configs/phase1_signed_face_boot_diagnostic_prereg_20260714.json)，无 launcher/Pod/
signal/training 权限。G05 继续 `Partial`。

### 2026-07-14 C2/D2 provenance-complete L1 source gate

v9 的只读证据复核发现旧 hard contract `dfc583d4...888a5` 没有绑定 positional/signed-face guidance
weight，因而旧 C/D checkpoint 即使有 outer launch claim，也不能只靠 checkpoint+相邻合同区分
`0.0/-0.4`。新训练 source `4467d79f1ed425a4263f0caaad2f661e1ec737ad` 把两项 post-Hydra
guidance 的 weight/command/bound 写入 schema-3 hard contract；checkpoint `infos` 另绑定非自引用的原子
launch-claim SHA，claim 覆盖 exact source、优化配方、host/GPU lane、seed、终档迭代和 claim directory
identity。Kit carb/TBB `16/16` 与 `useOmniJob=false` 也由启动后 runtime marker fail closed。

新的 [`signed-face C2/D2`](../DEFINITIONS.md) manifest/launcher 只包含 fresh seed3 的两条
`512 env × 25 update` L1。为遵守团队先铺满六卡的调度，C2 固定 Pod1 GPU1、D2 固定 Pod1 GPU2；每条
claim 前本卡必须空。host-wide Kit boot lock 串行，但 C2 `runtime_verified` 后继续训练，D2 可立即在
另一卡 boot/并发。两条 command 的 local device 都是 `cuda:0` 且 source/PYTHONPATH/runtime 相同；
physical GPU 是 outer execution lane，不进入优化配方。每个 `model_24.pt` 必须 finite、iter24、lineage1，
绑定各自相邻含 guidance weight 的 hard contract 与 outer claim；pair finalizer 要求两合同去掉该唯一
nested weight 后逐值相同。

manifest/launcher SHA 为 `785ad96d...9895` / `0fa25020...03ba`；专项测试 `28 passed`，source
launch-claim/thread-cap `28 passed`，reward/hard-contract override `58 passed`，仓内 `tests/` 回归见同次
实验卷宗。本任务没有 Pod/runtime/trainer/checkpoint；旧 v9 artifact 不采用，activation/judge/L2/第二
seed/stop-promote/真机全为 false。复现见
[运行手册](../operations/run_phase1_signed_face_cd_l1.md)和
[实验卷宗](../experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)。因此这里只是可审阅的 E1
source gate，G05 保持 `Partial`。

### 2026-07-14 C2 两层 outer 假拒绝与 v1r2 continuation source gate

C2 已在 Pod1 GPU1 按上述 v1 claim 自然到达 `model_24.pt`；final log/hard contract/checkpoint SHA 为
`abffd457...6dc3` / `83f47ae6...2772` / `dbbc7a28...6f6`，canonical outer claim 为
`37fe2443...86e5`。trainer hard contract 正确写出 exact float
`mount_normal_sign_per_clip=[1.0,-1.0]`，但 v1 outer verifier 用整数 `[1,-1]` 作为 strict-type
期望，导致合法合同在 post-boot 被假拒绝。旧 `runtime_verified`、failure、terminal result 均 absent；
不得补造旧 runtime sidecar，也不得重跑 C2。

冻结 [`v1r1`](../DEFINITIONS.md) 接受 exact float 并拒绝 bool/int，却又错误要求 trainer compact
`question_bank` 直含第六个 `physics_contract_sha256`。source `4467d79` 实际只发
`sha256/schema_version/split/source_family_sha256/exact` 五键；physics 只在 exact NPZ metadata 与
source-family contract 内。v1r1 因而精确假拒绝合法 C2 hard contract。最后一次成功只读快照
`2026-07-13T22:32:07Z` 证明 v1r1 control/evidence/pair、D2 arm/exact run 全 absent 且没有
write/claim/launch；后续 SSH 状态 unknown，所以历史 absence 不构成当前授权。v1/v1r1 bytes 均冻结并
禁止运行。

新的 [`v1r2`](../DEFINITIONS.md) 是 D2-only source gate：manifest/launcher SHA 为
`4e202589...8c638` / `2b53c865...45a12`。它精确复现 v1r1 错误，严格接受 trainer 实际五键 shape，
再独立解析 exact NPZ `meta_json` 绑定 file/schema/split/source-family/physics SHA，并复算
source-family contract。伪造第六键、metadata drift 或 v1r1 evidence root/C2 receipt/pair receipt、
D2 arm/exact run 任一存在都 fail closed。C2 attestation 只能进入独立 `continuations/v1r2/`
no-clobber root：先写 content-bound absence receipt 并立即重查，再只允许未 claim 的 D2；preserved
C2 arm 不增加文件。实际写任何 v1r2 byte 前必须已完整通过 C2 terminal、exact bank、v1r1 假拒绝、
两张绑定 GPU 与 live absence 复核；首次 exclusive write 后的失败会保留 receipt 并永久阻断同 namespace
重试。v1r2 自有 JSON/NPZ metadata 的重复 key 也 fail closed。

外部 control 只接受 `scripts/ + configs/` 六文件 mini-tree，同时绑定 v1r2 与冻结 v1/v1r1 的
helper/manifest；safe relative paths 拒绝绝对/`..`/symlink。临时外部树 subprocess
`static-validate/plan` 已通过，缺任一文件、旧扁平布局和重复 JSON key 均失败；v1r2 专项攻击测试
`52 passed`，三代聚焦回归 `111 passed`，受支持的完整仓内 `tests/` 为 `934 passed, 10 skipped`。
本分支没有连接 Pod、安装 control、写 attestation 或启动 D2；因此这是 source gate，不是 C2/D2 成对 runtime
通过。activation/judge/L2/第二 seed/晋级/真机仍全部为 false，G05 保持 `Partial`。复现见
[操作文档](../operations/run_phase1_signed_face_cd_l1.md)与
[face-sign 卷宗](../experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。

### 2026-07-14 C2 零摩擦声明/发射不一致：D2 永久停止

v1r2 control 合入并精确安装后，`static-validate/plan` 通过；fresh `validate-runtime` 在任何
`continuations/v1r2`、attestation、D2 claim 或同名 training run 写入前，以
`hard contract is not 31/31 zero-friction` fail closed。现场相邻 hard contract SHA
`83f47ae6...2772` 的 31 个摩擦系数均为非零 PhysX 默认值。根因不是第三个 outer 假拒绝：冻结 manifest
声明 `zero_joint_friction=true`，但 C2 launch argv 与 optimization recipe 都没有
`task.plant.zero_joint_friction=true`，所以 trainer 正确记录了真实非零 plant。

C2 只保留为 nonconforming 根因证据，不能与正式零摩擦谱系混用；C2 不重跑，D2/v1r2 永久
`NO-LAUNCH`。下一次训练必须使用全新 C3/D3 namespace，并把同一零摩擦值同时绑定到 argv、optimization
recipe、claim 和 emitted hard contract。该新 source/runtime 门通过前，G05 保持 `Partial`。

### 2026-07-14 C3/D3 显式零摩擦 L1 source gate

全新 C3/D3 prereg 不复用 C2/D2 的 run、claim、environment receipt 或 artifact root。两条都是 fresh
seed3、`512 env × 25 update`；C3 guidance 为 `0`，D3 只把 signed-face guidance 改为 `-0.4`。manifest
和 launcher 要求 `task.plant.zero_joint_friction=true` 在每条 argv 恰好出现一次，并逐层绑定到 outer
optimization recipe、atomic claim、唯一 `ZERO_FRICTION_RUNTIME_OK` marker、31/31 finite-zero hard
contract 以及 terminal checkpoint replay。任一层不一致永久停止该 namespace，不自动 retry。

manifest/launcher SHA 为 `eefc8023...5dc2` / `19214890...a628`；`static-validate`、plan 与专项
`38 passed`，完整 `tests/` 为 `972 passed, 10 skipped`。本节仍只有 E1 source 证据：Pod runtime、
checkpoint、activation/judge/L2/第二 seed/晋级/真机全为 false。复现见
[实验卷宗](../experiments/2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md)与
[操作文档](../operations/run_phase1_signed_face_c3d3_l1.md)，G05 继续 `Partial`。

Runtime 随后在 Pod1 GPU1/GPU2 分别一次性启动 C3/D3。两条各自 hard marker 与 31/31
`ZERO_FRICTION_RUNTIME_OK` marker 唯一，均自然到 finite/iter24/lineage1 `model_24.pt`；terminal SHA 为
`8c579386...e8ef` / `ccb9933c...7f0e`。finalizer 证明两份 hard contract 除
`racket_guidance_reward.signed_face.weight` 外逐值相同，并发布 paired receipt SHA
`bb3cd749...bbde`。这把 L1 provenance 从 E1 提升到 E2，但仍无 K100 行为、L2、第二 seed 或晋级，
所以 G05 保持 `Partial` 且 C3/D3 禁止重跑。

### 2026-07-14 A0/A1 paired checkpoints complete，行为仍未判

非击球臂 A1 自然退出；A0 在 `model_1000.pt` 稳定写完后发生 terminal teardown hang。终档 embedded
iteration、finite、fresh lineage、相邻 hard SHA 和正式 failure regex 均先通过；精确 PGID `1811464`
对 `TERM` 20 秒无响应后，只向同一个单成员 PGID 发 `KILL`。冻结 v1r1 finalizer 随后验证 A0/A1
两臂 `model_200/500/1000.pt` 与唯一 `motion_imitation_body_names` 差异，paired result SHA 为
`30ba716b...d7d9`。该结果明确 `same_immutable_signed_paper_judged=false`，所以它只完成 checkpoint
证据，不回答“不模仿非击球臂是否更好”，也不授权第二 seed。见
[实验卷宗](../experiments/non_striking_arm_imitation_ablation_20260713.md)。G05 仍为 `Partial`。

### 2026-07-14 signed-face K100 paper runtime materialized

Pod1 使用 clean detached `748b6d5` source 和 exact rebound exam bank `60e1a7ad...d1ca` 完成单次
CPU-only consume。新 schedule 是 `100` 个唯一题、正反手各 `50`，file/semantic/question-order SHA 为
`f2777dcd...1ca` / `3ca4bdba...3365` / `09f778f2...bd0`；最后写出的 paper-only activation
file/content SHA 为 `e0125b0e...bb4` / `533beb03...3d8`。完整 receipt 见
[`phase1_signed_face_exam_k100_runtime_receipt_20260714.json`](../../configs/phase1_signed_face_exam_k100_runtime_receipt_20260714.json)。
它没有消费 checkpoint，也明确不授权 trainer、judge、L2、第二 seed、停止/晋级或部署；所以这不是
Isaac 行为结果，G05 保持 `Partial`。

### 2026-07-14 signed-face K100 checkpoint attestor source gate

paper 后的 generic [`checkpoint attestor`](../DEFINITIONS.md) 已完成 source/static gate。它不自动挑“最新”
模型；每个候选 request 必须显式绑定 checkpoint path/bytes/SHA、filename/embed iteration、相邻
`params/training_contract.json` SHA、fresh lineage integer `1` 和 producer claim canonical SHA。hard contract
必须逐类型保留 `deploy_parity_face179`、`shared_plus_y` 与 exact float
`mount_normal_sign_per_clip=[1.0,-1.0]`，并把 31-joint plant 字段算入 semantic SHA。

同一 request 还冻结 clean source commit/tree、judge/evaluator/scorer/schedule source closure、checkpoint 与
evaluator Python runtime fingerprints、MJCF bytes/SHA，以及 actual immutable schedule 和 activation 的
file/content SHA。consumer 必须直接读取 actual activation；旧 runtime receipt 摘要中的 integer `[1,-1]`
已由 versioned correction pointer 保留并降级，不能作为 signed numeric type 权威，也不能用 Python 数值
等价放行。

全部 no-write 检查通过后，consumer 才在由 checkpoint SHA 唯一导出的 no-clobber namespace 先写 evidence、
最后写 claim；partial 或已有 root 均不可复用。claim 仍标记
`attested_not_executed_no_decision`，judge/trainer/L2/第二 seed/stop-promote/formal score/部署/真机全 false。
路径通配/穿越、symlink ancestry、checkpoint/contract/request 中途替换、dangling namespace 与
evidence-only partial 都 fail closed。focused 攻击回归为 `21 passed`，rebase 后仓内 `tests/` 为
`956 passed, 9 skipped`，`py_compile` 与 `static-validate` rc0；没有 Pod 连接、runtime request、
checkpoint evidence/claim、judge 或训练结果。因此只是 E1 source gate，G05 继续 `Partial`。见
[实验](../experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-PAPER.md)与
[操作](../operations/run_phase1_signed_face_k100_checkpoint_attestor.md)。

### 2026-07-14 C3/D3 signed-face K100 paired execution source gate

最小 one-shot consumer 已 exact 绑定 paired L1 receipt `bb3cd749...bbde`、C3/D3 两份终档四键
（checkpoint/hard/producer claim/terminal）、generic checkpoint attestor 和同一 K100
schedule/activation。任一侧 attestation、actual float `[1.0,-1.0]`、独立 eval worktree/runtime、MJCF/plant、
env.yaml 或 no-clobber namespace 缺失都拒绝；现有 judge 只允许在 distinct empty GPU 上顺序运行，不写训练
run、不发 signal。focused `28 passed`，static/source-plan rc0；尚未 runtime attest/judge，因此仍无行为结果，
G05 保持 `Partial`。见[实验](../experiments/2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md)与
[操作](../operations/run_phase1_signed_face_c3d3_k100.md)。

### 2026-07-14 A2/B2 跨 Pod 热启动 L1 runtime source gate

全新 v2 consumer 把旧 plan-only A2/B2 收口为两条独立 one-shot 探索 L1：Pod1 GPU0=A2 对照、Pod2
GPU0=B2 signed-face guidance。两条都绑定同一父 checkpoint、clean source commit/tree、`512×25`、唯一
`task.plant.zero_joint_friction=true` argv、outer claim、runtime zero-friction marker、31/31 zero hard
contract、fresh namespace/empty GPU 和 no-retry；terminal 必须是 finite `model_13824.pt`、lineage0。
focused `27 passed`，static/plan rc0；actual-host 由两 Pod GPU0 UUID 而非 CLI 自报绑定；跨 Pod pair
finalizer 完整比较两 hard contracts，并把 current-only 值锁到预注册值。尚无 Pod runtime/checkpoint/行为结果，不授权 judge/L2/第二 seed/
晋级/真机，G05 保持 `Partial`。复现见[操作](../operations/run_phase1_signed_face_a2b2_l1.md)。

### 2026-07-14 动作专属轻量 YAML 队列 source gate

新增探索训练用的单一 YAML 入口：每条 job 把 action/motion、专属 train bank/immutable exam、source
commit、base recipe/唯一 delta、seed、训练预算、checkpoint milestone 和资源策略放在一起。默认
`plan/status/launch-next` 都是 dry-run；只有显式 simulation-only token 才能启动一条 `ready` job，
`blocked` 永不调度。六卡先各放一条再进入下一圈，Pod1/Pod2 容量分别固定为每卡 `4/3`。

探索入口只查 clean commit、必需资产存在、GPU 容量、重复 claim 与 Kit boot lock，不引入逐文件 SHA、
pip/import closure 或 receipt；三个 runner 入口固定为 canonical repo-relative 路径，ready placeholder
在 SSH 前拒绝，并用全局 scheduler flock 包住六卡重采样/round-robin 选槽/launch；GPU 占用按唯一
numeric PID 计数，拒绝 `nvidia-smi` 重复行导致的假满。正式晋级/Gate3 仍用严格合同。focused 测试与静态检查见
[操作文档](../operations/run_lean_training_queue.md)。当前示例全部 blocked，尚无 Pod/训练/行为结果，
G05 保持 `Partial`。

#### Fresh C 五机制 attempt-1 基础设施失败与 harness 修复

active queue 的五个 attempt-1 都在 Pod1 GPU0 创建 claim 并启动过子进程，但均为 0 update、
pre-marker rc1、无 model，PID/PGID 已退出。根因是 setup 只导出 `HOPE_WBT_PYTHONPATH`，旧 launcher 的
raw Python 没有收到 `PYTHONPATH`，触发 `ModuleNotFoundError: whole_body_tracking`；因此不能据此拒绝任何
机制。旧 namespace/log/claim 已保全并标为 `rejected`，五个完全同 recipe 的 `retry-v2` 是唯一允许的
基础设施重试。

harness 现让 doctor/trainer 共用 CUDA+source-first PYTHONPATH builder，在 `mkdir/claim` 前验证 clean exact
source、assets 和 exact `find_spec` origin；SSH 错误保留 phase/stdout/stderr，launcher 等第一个
`Learning iteration`。`doctor --live` 不写 run 状态，并明确不声称无 Kit Hydra compose；`fill` 由单个
scheduler 进程逐条 doctor→claim→first iteration→重采。focused `17 passed`；retry-v2 尚未启动，
无 checkpoint/行为结果，G05 继续 `Partial`。见
[实验](../experiments/2026-07/EXP-P1-FRESH-C-MECHANISM-ABLATION.md)与
[操作](../operations/run_lean_training_queue.md)。

只读 runtime doctor 随后在五个 retry-v2 分配槽全部返回 `DOCTOR_OK`：两 Pod source/assets/exact module
origin 通过，实际 child Python 为 `/workspace/hope_isaac_venv/bin/python`；live 六 GPU occupancy 为 0。
该命令没有创建 retry-v2 claim 或 trainer，Hydra compose 仍明确未运行，所以只把 harness 从 E1 提升为
module-runtime preflight 通过，不改变 G05 `Partial`。

随后单一 `fill` scheduler 把五条 retry-v2 依次铺到 Pod1 GPU0/1/2 与 Pod2 GPU0/1，并全部越过第一个
`Learning iteration`。2026-07-14T10:09:52Z 五条仍存活于 update `103–160/1001`，无 fatal；五份
`model_100.pt` 的 filename=embedded iteration、finite、相邻 schema-3 hard-contract SHA 与 fresh lineage
均通过。第六格 qdot-limit tail 已冻结为同 fresh-C 配方、weight `-5.0`/margin `0.85`，只授权单 seed
`+200` direction screen；同 source weight0 matched control 前不得作因果采用。G05 仍为 `Partial`。

#### Lean queue 发射前 P0 合同收紧

后续 source gate 把高频误发模式挡在 claim 前：recipe 的 `key/+key/++key` 统一后必须无重复，也不能覆盖
seed、预算、run、device、motion/bank 或 launch-claim 等 harness-owned key；Hydra flag、删除语法与
interpolation fail closed。`run_dir` 在整份 YAML 内唯一且不能位于 ready source 内，远端只能以不带 `-p`
的原子 `mkdir` 创建全新 namespace，已有目录/文件/symlink 都拒绝。

standalone doctor 和 launch 内置 doctor 现在都用同一条最终 override 向量执行
`train.py --cfg job --resolve`，在 claim 前完成 no-Kit Hydra compose。canonical claim content 绑定 source、
完整 caller argv、run name、预算/milestones、motion/bank/exam identity 和 Pod/GPU，其 digest 自动加入真实
trainer argv 的 `training_launch_claim_sha256`，claim envelope 同时保存完整执行 argv。focused
`19 passed`；本变更没有连接 Pod、启动 trainer 或产生 checkpoint/行为结果，且尚未增加 source-specific
asset/cache warmup 的 phase marker，因此 G05 保持 `Partial`。复现见
[操作](../operations/run_lean_training_queue.md)与
[实验](../experiments/2026-07/EXP-P1-FRESH-C-MECHANISM-ABLATION.md)。

五机制 `+500` 随后证明并非全失败：V1+V2 出现 composite `0.0893` / normal pass `0.268` 的强精度信号，
但 completion/fall 仍差；V2 单独格 completion `0.176` / pre-fall `0.751`，成为唯一可替换格。五份
`model_500.pt` 均 finite/contract/lineage 通过。qdot attempt-1 则在第 0 update 的 A3 URDF import 阶段
停住，900 秒后由 launcher 精确终止 PGID `323083`；没有 hard contract/model，故只作基础设施失败，
同 recipe 的全新 retry-v2 namespace 才允许重试。G05 仍为 `Partial`。

qdot retry-v2 随后在 Pod2 GPU2 通过 no-Kit compose 和真实 boot marker，并到 iter `79`；schema-2 claim
digest、96 项 `/proc` argv、`model_0.pt` finite、hard contract 与 fresh/claim lineage 全匹配，fatal `0`。
它只关闭“机制能否真正进入训练”的运行门；未到 `+200`，且同 source/seed 的 weight `0` 对照未运行，
不构成 reward 结论或晋级。

qdot weight `0` 同-source control 与 conditional-face `0/-0.4` 配对已在结果前写入 active YAML；三条均
只允许 Pod2 dispatch、seed3、4096×1001、save100 与 `+200/+500/+1000`，且各自绑定 exact source、
motion/bank/exam/plant。它们尚未 launch，不能把 machine prereg 写成 runtime 通过。

qdot control attempt-1 随后也在 iter `0` 的动态 URDF importer 返回前停住，无 contract/checkpoint；
成功 treatment 有相同 `libGLU`/malformed-axis warning 却完成 scene creation，因此 warning 与 qdot weight
均非差异根因。exact PGID `327651` 已保全日志后收口；只允许完全相同配方的 retry-v2 fresh namespace
再试一次，若同 phase 重复则停止 retry。G05 仍为 `Partial`。

lean harness 因此新增默认不执行的 `boot-warmup` source gate：只从预注册 job 派生 1 env×2 update、独立
claim/namespace、180 秒 boot 上限的非科学探针，reserved Pod 与科学确认 token 均不能授权它。聚焦 queue
suite `23 passed`；尚未在 Pod 执行，所以这里只是 E1，不证明 importer 已稳定。

随后 conditional source 在 Pod2 GPU1 的首个 warmup 自然退出并通过 2/2 updates；`model_0/1` finite，
embedded iter、schema3 contract、claim 与 fresh lineage 匹配，fatal0，且 claim 明确 `not_science`。这把该
source/host/GPU 的 cold-boot 门提升到 E2，但不构成 conditional control/treatment runtime 或行为证据。

通用 Kit launcher 同时加入默认 180 秒 stale-log watchdog：只在日志非空后跟踪 size/mtime，增长即重置；
marker 同 poll 优先，stale 时只收口已验证 pid=pgid 的自身进程组并写 sidecar，rc125；空日志保持 hard
timeout rc124，stat 异常 rc126。专项 `9 passed`、相关 retry/queue `50 passed`。这缩短卡死占槽时间，不是
URDF importer 根因修复或 runtime 成绩。

后续红队发现“spawn 时验证过 PGID”仍不足以授权数分钟后的 signal：PID/PGID 可能复用，且 TERM 后只看
leader 会把仍存活的 child 漏掉。watchdog 现把 leader PID=PGID、双读一致的 Linux starttime 与
`getpgid` 写入 adjacent evidence；TERM 前再验证并冻结整组成员，TERM wait 枚举整组，KILL 前只接受该成员
集合的 exact 子集。leader 在 TERM 前消失、PID reuse、读中漂移或新成员加入都 no-signal + manual-review；
leader 在 TERM 后退出、已绑定 child 残留则仍可按其原 PID/starttime/PGID 安全收口。该项只是 E1 source
安全门，未连接 Pod、未 signal 任何远端进程，也不改变训练配方，G05 仍为 `Partial`。

对应的 marker-priority 回归不再用 `sleep 0.5` 与一秒 watchdog 竞争调度；测试 shim 在第二次 marker probe
同步写入 marker，从而稳定覆盖“同一轮 watchdog 已到期时 marker 仍优先”的语义。生产 launcher、timeout
和 signal 路径均未改变，专项 launcher/process-group 回归为 `15 passed`。

并发发射的另一根因也已定位：`flock FILE command` 的 lock fd 被 detached trainer 继承，导致每 GPU
名义容量 3/4 实际只能再发第一条。lean harness 现由短命 controller 持 fd8，launcher child 显式
`8>&-`；新增 preferred-slot 容量/回退测试后 queue suite `24 passed`。现役 qdot 两条仍持旧锁，不做信号，
只让新发射使用修复。

资源边界随后切换为 Pod2-only：Pod1 的三条 Codex trainer 在 iter `792/782/743` 由 exact PGID `TERM`
收口，`model_700.pt` 与日志保留，未发 `KILL`；Pod1 复核无 Codex compute process并全部交给 Yikang。
active queue 的 `dispatch_pods: [pod2]` 是可执行合同，不依赖聊天记忆；旧 Pod1 claim 仍只读防重复。

### 2026-07-14 31 关节 qdot-limit hinge 源码门

VirtualBall reward stack 新增默认关闭的
[`qdot-limit hinge`](../DEFINITIONS.md#qdot-limit-hinge)：它计算
`mean(relu(abs(qd)/joint_velocity_limits - margin)^2)`，默认 `margin=0.85`，只接受非正
Reward 权重。实现直接消费 `robot.data.joint_vel` 与 `robot.data.joint_vel_limits` 的同一 31 关节
runtime order；任意关节子集/重排、零值、非有限上限或不同 environment 的 limit 漂移都会 fail closed，
不能退化成 `action_rate` 代理。

Hydra 的 `joint_velocity_limit_hinge_weight` / `joint_velocity_limit_hinge_margin` 已走 fail-loud
translation，并把 applied marker 写入启动日志。post-override weight、margin、公式、31-joint identity
order 与 runtime limit 来源同时进入 training hard contract；未来 outer launch claim 必须再绑定 exact
argv/manifest。reward-layer focused qdot tests 为 `21 passed`，整个 dependency-light override 文件为 `76 passed`；
actual reward math 的 Torch/Isaac-stub focused tests 为 `3 passed`，schema-3/launch-claim suite 为
`62 passed`。合计 qdot-focused selection 为 `30 passed`。这仍是 E1 source gate：没有 Pod
训练、runtime marker、checkpoint 或行为成绩，也没有授权第二 seed/judge/晋级，G05 保持 `Partial`。

### 2026-07-14 不逃离就绪区的固定预算 Reward 源码门

为区分“静态 signed-face 权重不对”与“拍面 Reward 在触点/拍速尚未就绪时争自由度”，新增默认关闭的
[`racket_face_conditional_guidance_weight`](../DEFINITIONS.md#conditional-face-guidance)。它只在 wide
strike window 内花固定成本；位置误差用 `9.5→7.5 cm`、完整拍速向量误差用 `1.0→0.5 m/s`
形成就绪度。未就绪时成本为 1，进入门后连续换成 `15°→180°` 的拍面误差分数。故就绪度提高不会
增加成本，策略也不能靠退出门来免罚；门外拍面梯度为零。输出仍在 `[0,1]`，非正 weight 的绝对值是
每个时间窗 step 的最大罚金预算。拍面对仍强制走 raw-A/target-A 的共享 `_face_pair`。

Hydra 只暴露一个非正 weight 轴；门宽和公式随 source 固定并进入 training hard contract。默认 off、
数值/compact support、无效 bounds、raw-A 接线和 override/hard-contract 负例已写入 focused tests。
机制 math/梯度/单调性专项 `6 passed`，override 全文件 `78 passed`，raw-A face suite `34 passed`，schema-3/
launch-claim `62 passed`，`py_compile` 与 `git diff --check` 通过。
源码与反向激励反例已合入 `main@61007e9`；当前没有 Pod runtime、checkpoint 或行为结果。后续
control/treatment 必须同新 source、同
seed/动作/bank/plant，只改 conditional weight `0/-0.4`，并按 `+200/+500/+1000` 早判。安全/self-hit/
fall/guard 退化不可由拍面收益补偿。G05 保持 `Partial`；见
[实验卷宗](../experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md)与
[训练操作](../operations/run_training.md)。

#### Conditional 正式启动推翻 1-env warmup 授权语义

同 source 的 Pod2 GPU1 `1 env × 2 updates` warmup 通过后，4096-env matched control 仍在 dynamic
URDF import 阶段停止日志增长：PID=PGID `332786`，没有 scene-created marker、hard contract、checkpoint
或 `Learning iteration`；claim digest 为 `caffd19e...da52`，实际 argv 与 claim 匹配。2026-07-14T12:10:05Z
只向该精确 PGID 发出 `TERM`，30 秒未退出后只向同一 PGID 发出 `KILL`；最终 pre-marker rc137，日志、
claim 和 launch sidecar 全部保留。serial fill 没有创建 treatment directory/claim，故没有第二个失败 arm。

这不是 conditional Reward 负结果，而是启动 harness 反例：1-env 只可回答最小 cache/import 路径，不能
代表 4096-env scene recipe。旧 source610 配对永久撤销；下一次 formal pair 必须换 fresh source/namespace，
让 stale-log watchdog 随 source 固定，并先以相同 source/GPU/4096 environments/scene recipe 的非科学
full-scene probe 越过 first iteration，同时写 trainer-owned runtime binding。上述能力和新运行证据闭合前，
G05 继续 `Partial`。

#### Trainer-owned binding、milestone attestor 与全规模非科学 probe

`main@0b632c7` 让 opted-in trainer 在真实 RSL log directory 选定后原子发布 `run_binding.json`，绑定
queue claim、source actual HEAD/clean、PID=PGID、`/proc` starttime/argv、物理 GPU 与预注册 milestones；
attestor 不再 glob/latest 猜 checkpoint，只沿 binding 打开 exact `model_N.pt`，复核 filename/embed iter、
floating/complex finite、相邻 schema-3 contract SHA、fresh lineage 与 launch claim，再 no-clobber 写 receipt。
旧 source 默认关闭，不能被追认。

1-env warmup 被真实 4096-env 卡死反例推翻后，`main@077e70c` 又加入独立 `full-scene-probe`：保留 exact
source/动作/bank/plant/GPU 和正式 `num_envs`，只改为 2 updates/save1；输出位于按 job/source/Pod/GPU/attempt
隔离的非科学 namespace，明确 `attestable=false/promotable=false`，专用确认词、reserved Pod、重复路径、
zero/placeholder source 与环境数漂移均 fail closed。source focused/扩展套件分别 `73/109 passed`。

最初的 Pod2 clean detached exact `077e70c` source 已准备 motion、train bank 与 K100 exam；它后续因缺
Git-ignored A3 tree 在 iter0 自然 fail closed。当前 conditional P1 pair 与 V1+V2×base-decel pair 已改绑
strict `main@caeb9ad` source；该 source 的 full-scene probe 已通过并显式解锁两对，但 scientific trainer
尚未点火。因此这里已有 E2 启动/终档运行证据、没有 Reward 行为结果，G05 维持 `Partial`。

full-scene probe 首次 dry/capacity preflight 又发现 execute 原先会复用 all-Pod `live_snapshot`：Pod2-only
probe 也会访问 reserved Pod1。P1.2 将该路径收窄为 selected dispatch Pod/GPU 唯一 PID 计数，未知输出、
空 dispatch 或达到容量均 fail closed，远端 fd8 二次容量检查不变；普通 fill 仍使用 all-Pod claim 快照。
这是 source gate 修复，不是 probe runtime 通过，修复合入前没有创建 run directory/claim/process。

同一轮控制面复核还发现 `fill` 每臂先通过独立 SSH 跑 standalone doctor，随后的 `_launch_script` 又在远端
短锁内重复完全相同的 source/assets/module/Hydra compose。第一遍既不保留容量也不写 claim，不能提供额外
安全，却多出一个网络与 compose 失败面。P1.3 删除 execute 路径的前置重复调用；每臂现在只剩一次原子
launch SSH，且内置 doctor 仍严格位于容量、namespace/claim 与 Kit spawn 之前。standalone doctor/dry-run
保持不变；这是 source/control-plane gate，不是 Pod runtime 或行为结果，G05 仍为 `Partial`。

P1.4 再关闭“看到 first iteration 就误当终档”的缺口。full-scene probe 现在预注册唯一内部
`milestones=[1]`，trainer 在独立非科学路径发布 claim/binding；source-pinned supervisor 只自然 `wait` 并
no-clobber 记录 normal rc 与 signal 的区别，绝不发 signal。selected-Pod-only finalizer 要求绑定的 trainer/
supervisor 和原 PGID 全部自然消失，再核对 current expected claim、scene→hard-contract→first-iteration phases、
fatal0、finite/embedded-iter1/fresh-lineage1 `model_1.pt`、adjacent schema-3 SHA、exact supervisor argv、
claim/source-asset receipt 与 motion/train-bank binding。still-live/orphan 不写终档；其他终态
失败写 immutable `unlock_authorized=false` 结果且禁止自动 retry。普通 milestone attestor 明确拒绝该
`attestable=false` binding，queue 也没有自动 unlock consumer。dependency-light 整合 focused `126 passed`；尚无用
合入 source 产生的远端 claim/binding/exit/result，因此这里只是 E1 source gate，G05 继续 `Partial`。

#### qdot matched pair `+500` mixed signal

同 source/seed 的 qdot weight `-5/0` 两份 `model_500.pt` 已通过 filename/embed iter、finite、fresh lineage、
hard-contract 与 queue-claim binding。updates `480–500` 配对均值中 treatment 的 raw qdot max、near-limit
fraction、torque saturation 分别下降 `16.4%/20.1%/35.5%`，pre/post fall 也改善；但 position pass 从
`0.418` 降到 `0.107`，position error 从 `0.219 m` 升到 `0.311 m`，exact composite 两边均为零。日志还缺
activation denominator 与 normalized/per-joint tail，所以只能判 mixed signal：不采用、不买第二 seed，
等待 terminal checkpoint 的 immutable judge；G05 不因此晋级。

matched control 后续自然完成到 `model_1000.pt` 并退出。该文件 SHA-256 为 `b6672869...12cb9`，
filename/embedded iter=`1000`、76 tensors/1,762,717 elements finite、fresh lineage `1`；内嵌与相邻
schema-3 contract SHA 同为 `25faa6f5...da12`，queue claim 为 `c73ac441...8a959`，fatal regex 为 `0`。
对应 treatment `model_1000.pt` 也为 iter `1000`、76 tensors/1,762,717 elements finite、fresh lineage `1`；
model/contract/claim SHA 分别为 `8814debb...556e` / `3f6a532a...9091` / `3910e3e2...8fb6`，fatal `0`。
updates `980–1000` 的 21 点均值已出现晚熟翻转：treatment/control 的 position pass=`0.878/0.593`、
error=`0.0474/0.0962 m`、signed composite=`0.310/0.146`、virtual return=`0.454/0.265`，而 fall 与
completion 基本持平。因此停止低剂量/interaction 扩展，把 `-5` 保留为晚熟候选；同题 immutable
MuJoCo/vendor judge 尚未执行，G05 仍不晋级。

#### P1 full-scene probe 暴露 ignored A3 source closure 缺口

Pod2 首个 4096-environment P1 probe 在 `Learning iteration` 前自然 `rc=1`：exact detached
`077e70c` source 缺少 Git 忽略的 `assets/agibot_a3/urdf/model.urdf`，因此没有 hard contract、checkpoint 或
Reward 结论。archive donor 的既有接受树为 46 regular files、15,378,264 bytes、canonical SHA
`0137f59b...26c6`；其中 URDF 实际闭包有 43 个唯一 mesh 引用。source `git status` clean 不能证明 ignored
runtime asset 存在。

P1.4 source gate 因此让 YAML source 显式绑定 target/donor/commit/完整 tree 合同；新增 selected-Pod-only
`prepare-source-assets`，在 source 无 trainer 时用 source-specific lock、source 外 no-clobber staging、
`renameat2(RENAME_NOREPLACE)` 与 no-clobber receipt 水合。声明者的 doctor 在 Hydra/run-dir/claim/Kit 前
重算 donor/target、43/43 URDF closure、Git-ignore 并消费 exact receipt；science claim 自动绑定完整 source
mapping。旧行不声明时兼容。该 source-gate 提交当时不远程水合、不重发 probe、不改变 blocked 状态；
后续 strict caeb 结果见下文。G05 保持 `Partial`。

#### Pod2-only pre-probe 发射闩

P1 source 现改绑 exact `main@caeb9ad` checkout。
[`launch_authorized=false`](../DEFINITIONS.md#launch-authorized) 时 `fill/launch-next` 会在任何
SSH 前拒绝；status/doctor 与 probe 前置门只读取 `dispatch_pods=[pod2]`，不再访问 reserved Pod1。历史七条
ready 行同步改为 complete/rejected，新 conditional 与 V1+V2×base-decel 两对预分 Pod2 GPU1/GPU2。strict
receipt 通过后，两份队列已显式切为 `launch_authorized=true`、四条科学行 ready；尚未点火或产生科学
checkpoint，G05 继续 `Partial`。

#### Full-scene probe P1.5 终档诚实门

P1.5 收口首个 probe 的“短跑结束但没有 supervisor exit receipt”问题。launcher 现在只在精确 PGID 已按
既有 identity helper 收口后写 `pre_marker_exit/watchdog_error/stale_timeout/boot_timeout` 终态；finalizer 可把
该证据冻结为 **failure-only** 结果，绝不解锁或自动 retry。普通 exit-receipt 路径新增实际 scene telemetry：
`num_envs` 必须等于 claim 的 4096、物理球开关与 `pb_ball/pb_table/pb_table_visual` 必须都真实存在；claim/
hard contract 还必须分别证明 `deploy_parity_face179` 与 31/31 PhysX 零摩擦。schema-3 正式 validator 从
claim-bound clean checkout 的 dependency-light `training_contract.py` 直接载入，禁止经过会启动 Kit/Omni 的
package `__init__`。PID 已复用只证明原 identity 已退出，仍由双扫描 PGID closure 阻止 orphan；并发 finalizer
仍以 atomic no-replace 胜者为准，只接受 byte-identical 重放。增量 focused `100 passed`；源码门本身不等于
Pod runtime，G05 继续 `Partial`。

旧 `main@c7e1a90` 随后完成一次非科学基础设施 canary。其 `probe_result.json` 内容 SHA-256 为
`02780b52df27255eea096f34dda9a26e806ae3a196c233a46a2af1cde16c4186`，finite `model_1.pt` SHA-256 为
`a813ea9ba8c058cf5ed2f9a9a8f8fe3b95ec0903cd3702831b99736736738e68`，相邻 hard-contract SHA-256 为
`c39cf1ae4bd99aa5ddce2a4c6c51cfd3858eba4884baeb369d5fdb1cf88df838`；76 个 tensor 的
1,762,715 个浮点元素全 finite，fatal 命中为 0，trainer/supervisor 的原 PGID 自然为空。hard contract 也
独立通过正式 schema-3 validator，但 c7 `probe_result.json` 的 `unlock_authorized=true` 只属于旧终档语义；
它没有受 P1.5 结果绑定证明实际 4096 environments、物理球与 `pb_ball/pb_table/pb_table_visual`，不能追认
或解锁科学训练。

clean exact `main@caeb9ad` 随后用全新 attempt `caeb_strict_terminal_pod2_gpu1_a1` 通过严格门。result/claim/
model/hard-contract SHA-256 分别为
`0d03bd0305a56e56440b14e1f41278a26c0cad3a84cc1245325faed1ef29b1d1`、
`7437db488d8aa062aba8de91fb517362cc609a81900f0e953f80e15174c36ad5`、
`e1b79d142c13bc2df513b2a7311fbeb7b610fc64047e095c1a54c76571fe3106`、
`c39cf1ae4bd99aa5ddce2a4c6c51cfd3858eba4884baeb369d5fdb1cf88df838`。result 绑定 actual
`num_envs=4096`、`physical_ball=true`、三实体全存在、76 个 tensor / 1,762,715 个浮点元素全 finite、
fatal0、自然空 PGID 与 clean caeb source，故 `unlock_authorized=true`。该 receipt 已被显式队列变更消费，
两对当时均变为 ready。probe 仍 `not_science=true / attestable=false / promotable=false`，不产生 Reward
结论、不授权第二 seed、judge、晋级或部署；G05 保持 `Partial`。

显式 unlock 后，conditional control/treatment 已分别在 Pod2 GPU1/GPU2 越过 first iteration，PID=PGID
`357023/357679`；尚无 checkpoint 早判，不能形成 Reward 结论。随后 interaction control PID=PGID
`358331` 在 first iteration 前的 dynamic URDF import 报 `malloc(): invalid size (unsorted)`、`rc=134` 并
自然退出，treatment 未发射；claim/namespace 保全。这是新的启动基础设施失败，不是 V1+V2×base-decel
Reward 或行为失败，也不能把单边 attempt 记成 matched pair。旧 control 行已 rejected/no-relaunch；逐字
同配方 `control_retry_v2` 与从未 claim 的 treatment 均 ready，只允许同一 `fill --count 2` 事务先等 retry
first iteration 再发 treatment。该事务已按序完成：retry-v2 PID=PGID `359240`（Pod2 GPU1）和 treatment
PID=PGID `359872`（GPU2）均越过 first iteration，pair 现 live；尚无 checkpoint/早判。G05 仍为
`Partial`。
