# Third-Party Notices

This internal `main` branch keeps source/vendor layers needed for real training, planner, mocap, MuJoCo, and sim-to-real work. Public starter branches may expose only a subset.

| Material | Location | Policy |
| --- | --- | --- |
| Agibot A3 URDF packages | `agi/URDF/A3T2.5-URDF-std-pingpang/`, `agi/URDF/a3_t2d5/` | Vendor-provided A3 robot descriptions. Keep as tracked internal source while license/redistribution terms are confirmed before any public release. |
| Agibot A3 deploy payload | `agi/code_deployment/a3_deploy_example/`, `vendor_assets/agibot/a3_deploy_example_full/` | Tracked subset contains source/config/docs. Full runtime payload is private, ignored, and must be restored locally. |
| Agibot MuJoCo/AimRT materials | `agi/A3_MuJoCo_Sim/` | Tracked internal source/vendor drop for MuJoCo/AimRT parity work. Runtime dependency pinning and public redistribution remain undecided. |
| BeyondMimic whole_body_tracking | `hope_training/whole_body_tracking/` | Internal fork/adaptation of the BeyondMimic training scaffold for HOPE/A3. Keep provenance in README and gate docs when importing upstream changes. |
| ChingMu/VRPN mocap client | `hope_ws/src/vrpn_mocap/` | ROS 2 VRPN/mocap integration source kept for internal bringup. External package provenance should be recorded when promoted beyond this repo copy. |
| TTRL reference | `external_repos/TTRL-ICRA2026/` | Ignored auto-synced reference clone, not a pinned dependency. Record source commit in gate docs when extracting ideas, config, code, or results. |
| Purdue PACE table/net visual asset | `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/table_usd/` | Small tracked runtime visual asset; its local license file records the Purdue PACE MIT license. |

Generated logs, checkpoints, ONNX/RKNN/TensorRT engines, WandB caches, motion `.npz` outputs, and copied Isaac robot assets remain ignored unless the team explicitly adopts an artifact system such as Git LFS.
