# HOPE Developer And Agent Start Here

This is the first file to read before changing code, moving assets, or assigning work. The repository is in an early reimplementation stage: it already contains reference documents, initial ROS planner code, calibration data, Agibot support materials, and training/deployment scaffolding, but the full real-to-sim-to-real loop is not complete.

## Project Goal

Build a HITTER-compatible humanoid table-tennis stack on Agibot A3:

1. Reproduce the useful HITTER system contract: model-based ball planning plus RL whole-body control.
2. Make the implementation work with the actual A3, ChingMu/VRPN motion capture, MuJoCo, Isaac, and Agibot deployment materials.
3. Use the baseline to identify and attack blind spots such as spin, double bounce, short/deep balls, serving, and opponent adaptation.

The goal is not to blindly clone every HITTER detail. The goal is to preserve the interface and evaluation discipline while adapting the implementation to this hardware and improving where the paper is weak.

## Current Gate Status

| Gate | Status | Primary Doc |
| --- | --- | --- |
| G00 Materials and harness | Partial | [G00](gates/G00_materials_and_harness.md) |
| G01 Real preparation | Partial | [G01](gates/G01_real_preparation.md) |
| G02 Data acquisition | Partial | [G02](gates/G02_data_acquisition.md) |
| G03 Data processing and physics calibration | Partial | [G03](gates/G03_data_processing_and_physics_calibration.md) |
| G04 Sim modeling in MuJoCo and Isaac | Partial | [G04](gates/G04_sim_modeling_mujoco_isaac.md) |
| G05 Isaac training first loop | Partial | [G05](gates/G05_isaac_training_first_loop.md) |
| G06 Isaac-to-MuJoCo parity | Not started | [G06](gates/G06_isaac_to_mujoco.md) |
| G07 MuJoCo-to-real deployment | Not complete | [G07](gates/G07_mujoco_to_real.md) |
| G08 Blind-spot improvements | Research track | [G08](gates/G08_blind_spot_improvements.md) |

Status labels:

- `Done`: passing verification exists in this repo.
- `Partial`: materials or code exist, but the gate is not fully verified.
- `Not complete`: a scaffold exists, but no accepted loop has been demonstrated.
- `Not started`: no meaningful implementation path exists yet.
- `Research track`: intentionally open-ended improvement work after the baseline.

## How To Start A Specific Task

Do not read every setup document before doing a specific job. Use this task-first path:

1. Pick the relevant gate from the table above.
2. Read only that gate doc, especially `Current State`, `Acceptance Criteria`, and `Next Steps`.
3. Open the operation doc linked by that gate.
4. Follow the `Task Setup` or setup section inside that operation doc.
5. Open interface or asset docs only when the operation doc links to them or when you are changing that contract.

For broad onboarding, read [PROJECT_MAP.md](PROJECT_MAP.md) and [DEFINITIONS.md](DEFINITIONS.md). For folder or asset-policy changes, read [ASSET_POLICY.md](ASSET_POLICY.md).

## Implementation Principles

Use first-principles gates instead of copying a paper or vendor example blindly:

1. Define the contract before optimizing the implementation.
2. Keep one source of truth for frames, joint order, policy observations, actions, and asset paths.
3. Measure or verify before tuning.
4. Separate source, curated data, runtime assets, and external references.
5. Prefer small reproducible gate checks over large undocumented demos.
6. Treat HITTER as the baseline system contract, not as a restriction on improvements.
7. Do not advance real-hardware risk faster than the safety documentation and dry-runs.

## Documentation Update Rule

Every human or agent must keep the docs in sync with the work.

Update docs in the same branch when any of these happen:

1. A gate target changes.
2. A gate status changes.
3. A new blocker is found or removed.
4. A new folder or artifact type is added.
5. An interface changes: frame, topic, message, joint order, observation, action, model format, or runtime asset path.
6. A setup, build, test, training, or deployment command changes.
7. A local-only asset becomes required to reproduce a result.

Minimum update locations:

- Update the relevant `docs/gates/G*.md` file.
- Update [PROJECT_MAP.md](PROJECT_MAP.md) if folder ownership changes.
- Update [ASSET_POLICY.md](ASSET_POLICY.md) if git, ignored, LFS, or submodule policy changes.
- Update [PROGRESS.md](PROGRESS.md) with a short dated entry.

Do not hide project state in chat history. If a future contributor or agent needs it, write it into the repo.

## Fast Entry Points

- End-to-end, machine-specific runbook (environment creation, GMR/GVHMR motion pipeline, A3 asset prep, training, deploy): [reimplement.md](../reimplement.md). The `operations/*` docs are the curated per-task views; `reimplement.md` is the long-form narrative source they defer to (e.g. "Step 12.7").
- Environment setup is task-specific. Start from the operation doc for your task; use [operations/setup_environments.md](operations/setup_environments.md) only as a reference matrix.
- Ignored/local assets that must be copied manually are summarized in [operations/setup_local_sync.md](operations/setup_local_sync.md), but each operation doc should list the assets it needs locally.
- Auto-sync ignored external references before using them: `scripts/sync_external_repos.sh`.
- Planner tests: see [operations/build_and_test.md](operations/build_and_test.md).
- Mocap bringup: see [operations/run_mocap.md](operations/run_mocap.md).
- Planner runtime: see [operations/run_planner.md](operations/run_planner.md).
- Isaac training: see [operations/run_training.md](operations/run_training.md).
- A3 deploy dry-run: see [operations/run_deploy_dryrun.md](operations/run_deploy_dryrun.md).
- Local assets and sync: see [operations/setup_local_sync.md](operations/setup_local_sync.md).

## Current Known Environment Limits

- The project runs on Linux with ROS 2 Jazzy. The `hope_ws` `colcon` build has not yet been independently verified in this harness shell, so run and verify it inside the ROS environment ([operations/build_and_test.md](operations/build_and_test.md)).
- A fresh `git clone` is intentionally **not** self-contained: `external_repos/` and `vendor_assets/` do not exist after checkout, and the `hope_training/GMR` / `hope_training/GVHMR` clones, reference motions, and binary model artifacts are git-ignored. Recreate them on demand — `scripts/sync_external_repos.sh` for TTRL, and the manual restore steps in [operations/setup_local_sync.md](operations/setup_local_sync.md) for everything else.
- Agibot runtime assets under `vendor_assets/` are local and ignored by git (the full A3 deploy payload is a private ~1.7 GB vendor handoff).
- TTRL is an ignored auto-synced reference under `external_repos/`, updated through `scripts/sync_external_repos.sh`; it is intentionally not pinned unless a future gate promotes it to a dependency.
