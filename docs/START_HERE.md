# HOPE Developer And Agent Start Here

This is the first file to read before changing code, moving assets, or assigning work. The repository has walked the Isaac → MuJoCo → real-A3 chain once for the policy side (first sim-to-real transfer of the unified swing policy on 2026-07-02, forehand only, with real behavior matching MuJoCo), but the loop is not closed: the perception chain (mocap → planner → deploy runner) is not yet bridged, there is no accepted quality baseline, and the data-collection/physics-calibration phase has not run.

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
| G06 Isaac-to-MuJoCo parity | Partial | [G06](gates/G06_isaac_to_mujoco.md) |
| G07 MuJoCo-to-real deployment | Partial | [G07](gates/G07_mujoco_to_real.md) |
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

**Before starting ANY work item, claim it in [NOW.md](NOW.md)** (owner + branch). NOW.md is the
short-horizon board (who is doing what right now); the long-horizon roadmap is
[gates/G08_blind_spot_improvements.md](gates/G08_blind_spot_improvements.md).

## Reimplementation Rhythm

`reimplement.md` is the long-form supplemental runbook, not the primary setup path or a separate project plan. Use it through the gate docs:

1. Find the gate that matches the work.
2. Read the gate doc and operation doc first.
3. Use the referenced `reimplement.md` step for command detail.
4. Verify the gate's acceptance criteria.
5. Update the gate doc, `docs/PROGRESS.md`, and any touched interface/operation docs in the same branch.

If a phase/step in `reimplement.md` conflicts with a gate doc, treat the gate doc as current and update both files before continuing.

## Implementation Principles

Use first-principles gates instead of copying a paper or vendor example blindly:

1. Define the contract before optimizing the implementation.
2. Keep one source of truth for frames, joint order, policy observations, actions, and asset paths.
3. Measure or verify before tuning.
4. Separate source, curated data, runtime assets, and external references.
5. Prefer small reproducible gate checks over large undocumented demos.
6. Treat HITTER as the baseline system contract, not as a restriction on improvements.
7. Do not advance real-hardware risk faster than the safety documentation and dry-runs.
8. **Grade your sources.** Papers and our own measurements are primary and may drive decisions;
   relayed verbal information (author chats, hallway advice) is secondary — record it dated and
   marked 转述/未验证, use it to corroborate or to form hypotheses, and when it conflicts with a
   primary source, the primary source wins and the conflict is written down (see G08 for the
   pattern).

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

- New computer or agent startup: use this file as the index, then go to the operation doc for the task. For Isaac training, use [operations/run_training.md](operations/run_training.md); for ignored/private assets missing from a fresh clone, use [operations/setup_local_sync.md](operations/setup_local_sync.md). Use [reimplement.md](../reimplement.md) only as the long-form runbook when a gate or operation doc points to a specific step.
- End-to-end historical runbook (environment creation, GMR/GVHMR motion pipeline, A3 asset prep, training, deploy): [reimplement.md](../reimplement.md). The `operations/*` docs are the curated per-task source of truth; use `reimplement.md` only for supplemental command detail when an operation or gate doc cites a specific step (e.g. "Step 12.7").
- Environment setup is task-specific. Start from the operation doc for your task; use [operations/setup_environments.md](operations/setup_environments.md) only as a reference matrix.
- Ignored/local assets that must be copied manually are summarized in [operations/setup_local_sync.md](operations/setup_local_sync.md), but each operation doc should list the assets it needs locally.
- Auto-sync ignored external references before using them: `scripts/sync_external_repos.sh`.
- Planner tests: see [operations/build_and_test.md](operations/build_and_test.md).
- Mocap bringup: see [operations/run_mocap.md](operations/run_mocap.md).
- Planner runtime: see [operations/run_planner.md](operations/run_planner.md).
- Isaac training: see [operations/run_training.md](operations/run_training.md).
- Shared RunPod GPU training (team pod, per-user folders, smoke suite): see [operations/run_on_runpod.md](operations/run_on_runpod.md).
- Video-to-motion references for training: restore ignored GVHMR/GMR assets via [operations/setup_local_sync.md](operations/setup_local_sync.md), then follow [reimplement.md](../reimplement.md) steps 9-12 to produce local `hope_training/motions/preprocessed/*.npz`; upload to WandB only when you need shared registry artifacts.
- A3 deploy dry-run: see [operations/run_deploy_dryrun.md](operations/run_deploy_dryrun.md).
- Local assets and sync: see [operations/setup_local_sync.md](operations/setup_local_sync.md).

## Current Known Environment Limits

- The project runs on Linux with ROS 2 Jazzy. The `hope_ws` `colcon` build has not yet been independently verified in this harness shell, so run and verify it inside the ROS environment ([operations/build_and_test.md](operations/build_and_test.md)).
- A fresh `git clone` is intentionally **not** self-contained: `external_repos/` and `vendor_assets/` do not exist after checkout, and the `hope_training/GMR` / `hope_training/GVHMR` clones, reference motions, and binary model artifacts are git-ignored. Recreate them on demand — `scripts/sync_external_repos.sh` for TTRL, and the manual restore steps in [operations/setup_local_sync.md](operations/setup_local_sync.md) for everything else.
- Agibot runtime assets under `vendor_assets/` are local and ignored by git (the full A3 deploy payload is a private ~1.7 GB vendor handoff).
- TTRL is an ignored auto-synced reference under `external_repos/`, updated through `scripts/sync_external_repos.sh`; it is intentionally not pinned unless a future gate promotes it to a dependency.
