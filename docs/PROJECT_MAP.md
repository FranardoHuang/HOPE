# Project Map

This file describes stable repository zones. Gates describe work over time; folders describe long-lived ownership and artifact types. Do not split folders only because a phase changes.

## Top-Level Zones

| Path | Role | Git Policy |
| --- | --- | --- |
| `docs/` | Living project map, gates, interfaces, and operations | tracked |
| `papers/` | Source papers used as technical ground truth | tracked when license allows |
| `mocap/` | Motion-capture setup, coordinates, and vendor notes | tracked |
| `scripts/` | Repo maintenance helpers for local sync and project hygiene | tracked |
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
| `hope_planner` | Ball estimation, trajectory prediction, calibration, and racket target planning |
| `vrpn_mocap` | ChingMu/VRPN motion-capture ROS 2 client |

Future packages should be added here when they are runtime ROS packages, for example `hope_a3_bridge` or `hope_policy_runtime`.

## Agibot Zone

| Path | Role |
| --- | --- |
| `agi/URDF/` | A3 robot descriptions and meshes |
| `agi/A3_MuJoCo_Sim/` | MuJoCo/AimRT simulation example and source |
| `agi/code_deployment/` | A3 deployment documents and source examples |
| `agi/code_deployment/a3_deploy_example/` | Tracked source/config subset of the Agibot deploy package |
| `vendor_assets/agibot/a3_deploy_example_full/` | Full local deploy package including heavy models, sysroots, and binaries |

## Training Zone

`hope_training/whole_body_tracking` is the current Isaac/BeyondMimic training entry. The A3 `TrackingFlat` and `HOPEPingPong` forehand pipeline can train and export ONNX on the copied A3 URDF asset. Treat this as pipeline viability, not as an accepted quality baseline; use [gates/G05_isaac_training_first_loop.md](gates/G05_isaac_training_first_loop.md) for acceptance status.

The same package now also contains a first-pass table-tennis physics/visualization task:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/`: motion-imitation and `HOPEPingPong` racket-target WBC training tasks.
- `source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/`: `HOPE-TableTennis-AgibotA3-v0`, a full HOPE-frame table/net/ball/A3 scene with drag/Magnus hooks, 400 Hz physics, observations, and placeholder rewards for future returner work.
- `source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/table_usd/`: tracked, small Purdue PACE table/net USD visual overlay used by the table-tennis scene; physics remains owned by the task's cuboid colliders.
- `scripts/play_table_tennis.py`: visualization/headless smoke runner for the table-tennis scene.

Additional training config:

- `hope_training/config/joint_order_agibot_a3.yaml`: current working 31-DOF A3 joint order for training/export alignment.
- `hope_training/whole_body_tracking/setup_train_env.sh`: source this inside the GPU/Isaac environment before running training/eval commands.

## External Reference Zone

`external_repos/TTRL-ICRA2026` is currently an ignored auto-synced local clone. Keep it current with:

```bash
scripts/sync_external_repos.sh
```

It is allowed to drift with upstream while it is only used as reading material. If it becomes a stable dependency, promote it to one of these:

1. Git submodule pinned to upstream if we only read it.
2. Fork plus submodule if we need to patch it.
3. Vendored subtree only if submodule workflow becomes a blocker.

Record that decision in [ASSET_POLICY.md](ASSET_POLICY.md) and the relevant gate doc.

## Folder Change Rule

Before moving or creating major folders, update this file and the affected gate doc. Folder moves should preserve runnable commands or include migration notes.
