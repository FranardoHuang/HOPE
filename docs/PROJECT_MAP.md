# Project Map

This file describes stable repository zones. Gates describe work over time; folders describe long-lived ownership and artifact types. Do not split folders only because a phase changes.

## Top-Level Zones

| Path | Role | Git Policy |
| --- | --- | --- |
| `docs/` | Living project map, gates, interfaces, and operations | tracked |
| `papers/` | Source papers used as technical ground truth | tracked when license allows |
| `mocap/` | Motion-capture setup, coordinates, and vendor notes | tracked |
| `scripts/` | Repo maintenance helpers for local sync, asset preparation, and project hygiene | tracked |
| `calib_bags/` | Small reference raw calibration recordings | tracked only for curated samples |
| `calib_csv/` | Processed calibration CSVs, chunks, and plots | tracked only for curated samples |
| `hope_ws/` | ROS 2 integration workspace | tracked source, ignored build/install/log |
| `hope_training/` | Isaac/BeyondMimic training scaffold | tracked source, ignored generated motions and logs |
| `agi/` | Agibot URDF, MuJoCo, deployment docs, and source support | tracked source and small runtime configs |
| `vendor_assets/` | Full local vendor/runtime payloads: models, sysroots, binaries | ignored |
| `external_repos/` | Auto-synced local reference clones not yet promoted to formal dependencies | ignored |

## ROS Workspace

`hope_ws` should remain one integration workspace with multiple packages:

| Package | Role |
| --- | --- |
| `hope_bringup` | World frame, launch files, relays, and integration config |
| `hope_msgs` | Shared ROS messages |
| `hope_planner` | Ball estimation, trajectory prediction, calibration, and racket target planning; includes the `planner_imitate` scripted-target node feeding `/racket/command` |
| `hope_wbc_runner` | Staged, safety-gated ONNX WBC runner (dry_run/shadow/hardware, `/hope/estop`) consuming `/racket/command`; legacy 180-D companion — the shipping deploy path is the C++ runner in `agi/a3_deploy_example` |
| `vrpn_mocap` | ChingMu/VRPN motion-capture ROS 2 client |

Future packages should be added here when they are runtime ROS packages, for example `hope_a3_bridge`.

## Agibot Zone

| Path | Role |
| --- | --- |
| `agi/URDF/` | A3 robot descriptions and meshes |
| `agi/A3_MuJoCo_Sim/` | MuJoCo/AimRT simulation example and source |
| `agi/a3_deploy_example/` | **Active tracked ping-pong deploy tree**: Route A C++ runner `a3_deploy_onnx_ref_pingpong` (reuses AGI `A3AimrtBackend`/`A3PolicyDriver`), build/packaging scripts, and the PINGPONG_*/MUJOCO_VALIDATION/HARDWARE_BRINGUP runbooks — the chain that walked sim-to-real on 2026-07-02 |
| `agi/code_deployment/` | A3 deployment documents and source examples (vendor reference) |
| `agi/code_deployment/a3_deploy_example/` | Older tracked vendor subset of the Agibot deploy package (no ping-pong front-end) |
| `vendor_assets/agibot/a3_deploy_example_full/` | Full local deploy package including heavy models, sysroots, and binaries |

## Training Zone

`hope_training/whole_body_tracking` is the current Isaac/BeyondMimic training entry. The unified
forehand+backhand `HOPEPingPongDeployParity` task (175-D deploy-parity actor obs; the 2026-07-01
`train.py` blocker is fixed) produced the first hardware-deployed swing policy
(`model_p4_deployparity.onnx`, 2026-07-02, forehand only). Sim2sim validation lives in
`scripts/mujoco_eval_onnx.py`; the observation-contract references are
`scripts/realsensor_obs_reference.py` / `scripts/verify_realsensor.py` /
`REALSENSOR_OBS_REDESIGN.md`. There is still no accepted quality baseline; use
[gates/G05_isaac_training_first_loop.md](gates/G05_isaac_training_first_loop.md) for acceptance
status.

Stable cross-stack interface documents live under `docs/interfaces/`. In
particular, `policy_observation_action.md` owns tensor/export semantics and
`racket_contact_geometry.md` owns the URDF/MJCF racket site, rigid-point
velocity and versioned physical-contact migration. Do not duplicate those
constants in a gate narrative as a second source of truth.

The same package now also contains a first-pass table-tennis physics/visualization task:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/`: motion-imitation and `HOPEPingPong` racket-target WBC training tasks.
- `source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/`: `HOPE-TableTennis-AgibotA3-v0`, a full HOPE-frame table/net/ball/A3 scene with drag/Magnus hooks, 400 Hz physics, observations, and placeholder rewards for future returner work.
- `source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/table_usd/`: tracked, small Purdue PACE table/net USD visual overlay used by the table-tennis scene; physics remains owned by the task's cuboid colliders.
- `source/whole_body_tracking/whole_body_tracking/assets/`: package-local asset path helper plus ignored copied/generated robot assets; restore `assets/agibot_a3/` from tracked Agibot URDF materials per `docs/operations/setup_local_sync.md`.
- `source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/`: ignored generated Isaac asset rebuilt from `agi/URDF/A3T2.5-URDF-std-pingpang/` with `scripts/prepare_a3_isaac_asset.py`.
- `scripts/play_table_tennis.py`: visualization/headless smoke runner for the table-tennis scene.

Ball-physics fitting (2026-07-03):

- `hope_training/ball_physics_fit/`: the venue ball-physics fitting pipeline of record
  (C3D extraction → QA → segmentation → ordered fits → F1–F8 falsification battery in
  `falsify/` → held-out validation → two-horizon landing `predict_check.py`). Data root
  via `BALLFIT_DATA_ROOT`; deps in `requirements-ballfit.txt` (`c3d` only for extraction).
  Fitted constants: `configs/ball_physics_venue.yaml`; report: `docs/ball_physics_fit_report.md`.

Additional training config:

- `hope_training/config/joint_order_agibot_a3.yaml`: current working 31-DOF A3 joint order for training/export alignment.
- `hope_training/whole_body_tracking/setup_train_env.sh`: source this inside the GPU/Isaac environment before running training/eval commands.
- `hope_training/whole_body_tracking/scripts/audit_runpod_terminal_runs.py`: read-only inventory of
  historical terminal checkpoints and judge sidecars.  It prints explicit commands but never
  launches, deletes or resumes a run.
- `hope_training/whole_body_tracking/scripts/termination_contract.py`: simulator-independent,
  fail-closed specification for freezing the termination/timing fields of a saved
  `params/env.yaml`.  It is not yet wired into the production evaluator.
- `hope_training/whole_body_tracking/scripts/virtual_return_scorer.py`: NumPy specification of the
  Isaac RK4 virtual-return metric, including ball-centre table contact.  The production venue and
  BankExam adapters remain pending.
- `docs/RUNPOD_TRAINING_AUDIT_2026-07-10.md` and `docs/PHASE1_SUMMARY_2026-07-10.md`: historical
  Phase-1 evidence and candidate closeout design.  They are not current launch authorization;
  current ordering and acceptance authority stay in `docs/NOW.md` and the gate documents.

## External Reference Zone

`external_repos/TTRL-ICRA2026` is currently an ignored auto-synced local clone. Keep it current with:

```bash
scripts/sync_external_repos.sh
```

`external_repos/IsaacLab` is an ignored local Isaac Lab source checkout used by the G05 training
environment when a pre-provisioned Isaac Lab checkout is absent. Record the tag/commit in G05 before
using it for a reproducible training result.

It is allowed to drift with upstream while it is only used as reading material. If it becomes a stable dependency, promote it to one of these:

1. Git submodule pinned to upstream if we only read it.
2. Fork plus submodule if we need to patch it.
3. Vendored subtree only if submodule workflow becomes a blocker.

Record that decision in [ASSET_POLICY.md](ASSET_POLICY.md) and the relevant gate doc.

## Folder Change Rule

Before moving or creating major folders, update this file and the affected gate doc. Folder moves should preserve runnable commands or include migration notes.
