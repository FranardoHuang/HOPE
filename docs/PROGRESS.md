# Progress Log

Use this file for short project-state updates that future humans and agents need to see. Keep detailed reasoning in the relevant gate doc.

## 2026-06-25

- Merged `origin/train_1` into `main` and updated the docs/reimplementation rhythm so `reimplement.md` is gate-indexed rather than a separate phase plan. G04 now records the new `HOPE-TableTennis-AgibotA3-v0` Isaac scene, and G05 records the updated `HOPEPingPong` exact-strike metrics, reward/target defaults, source-provenance logging, and success-gated reference perturbation.
- Preserved source/config/test changes while excluding generated or local artifacts (`.codex-tmp/`, `.vscode/`, generated `*.onnx`) under the asset policy. Scrubbed private training paths and WandB identities from `reimplement.md`; machine-specific values belong in `setup_train_env.local.sh`.

## 2026-06-24

- Integrated `origin/jiayi` commit `c951d9d` into this harness branch while preserving the open-source placeholders and A3 support-material docs. The branch now includes `reference_perturbed` racket target sampling for `HOPEPingPong`, training override plumbing for the new parameters, and Avatar-Pro relay `ball_tracking_mode` support (`rigid_body` preferred, `auto` fallback).

## 2026-06-23

- Open-source documentation pass for this branch: rewrote `README.md` (About-this-branch, Repository Layout, Quickstart, Assumptions & Limitations; corrected A3 31/29-DOF and ChingMu/VRPN framing), indexed `reimplement.md` from `START_HERE`, and filled env-setup gaps across `docs/operations/*` (environment creation, GMR/GVHMR + SMPL-X/checkpoint restore, WandB team-vs-org + offline `logger=tensorboard`, the `Live/...` telemetry and the reward-std `strike_success=0` fix) and `docs/interfaces/*` (31-DOF list, 31-vs-29 deploy, real actor/critic observations + action dims, world-Z frame, RacketCommand fields).
- Scrubbed maintainer-private values for public release: `setup_train_env.sh` now reads overridable `HOPE_ISAAC_*` paths and placeholder WandB identity (real values via a git-ignored `setup_train_env.local.sh`); added `ros-jazzy-vrpn-mocap` + `python3-vcstool` to `Dockerfile.hope-ros2-jazzy`.
- Integrated A3 Isaac/BeyondMimic training updates (through commit `42489cd`): working joint-order YAML, updated A3 robot config, deploy-transcribed PD/action-scale values, HOPEPingPong task updates, train/play fixes, `setup_train_env.sh`, richer WandB/live metrics, and in-container ONNX export support.
- Added `scripts/sync_external_repos.sh` so TTRL remains an auto-synced ignored reference instead of a pinned dependency; docs now require recording the TTRL source commit when material is extracted from it.

## 2026-06-22

- Added HITTER paper under `papers/2508.21043v2.pdf`.
- Added ChingMu VRPN ROS 2 package under `hope_ws/src/vrpn_mocap` and supplied a minimal ROS 2 `package.xml`.
- Added tracked source/config subset of Agibot `a3_deploy_example` under `agi/code_deployment/a3_deploy_example`.
- Moved the complete Agibot deploy payload to ignored local storage under `vendor_assets/agibot/a3_deploy_example_full`.
- Added TTRL as an ignored local reference under `external_repos/TTRL-ICRA2026`.
- Removed root `tmp/` after moving useful materials.
- Verified `hope_planner` package tests from `hope_ws/src/hope_planner`: 20 passed.
- Could not run ROS workspace build in the current shell because `colcon` is not installed.
