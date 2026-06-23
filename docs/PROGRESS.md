# Progress Log

Use this file for short project-state updates that future humans and agents need to see. Keep detailed reasoning in the relevant gate doc.

## 2026-06-23

- Pulled `origin/jiayi` updates through `42489cd`, which added A3 Isaac/BeyondMimic training progress: working joint-order YAML, updated A3 robot config, deploy-transcribed PD/action-scale values, HOPEPingPong task updates, train/play fixes, `setup_train_env.sh`, richer WandB/live metrics, and in-container ONNX export support.
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
