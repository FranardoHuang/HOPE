# Documentation Index

This is the one-stop router after [START_HERE.md](START_HERE.md). Read the smallest set for the
task at hand; do not treat a historical report or chat transcript as current launch authority.

## Authority

1. [NOW.md](NOW.md) is the main-only current work board and priority queue.
2. [gates/](gates/) own acceptance state and blockers.
3. [experiments/](experiments/README.md) own individual hypotheses, preregistration, runs and
   decisions.
4. [interfaces/](interfaces/) own cross-component contracts.
5. [operations/](operations/) own reproducible commands.
6. [PROGRESS.md](PROGRESS.md) and [TIMELINE.md](TIMELINE.md) record dated evidence and mainline
   history; they do not authorize a run by themselves.

## Route By Task

| Task | Read first | Then read |
| --- | --- | --- |
| Current roadmap, owners and next checkpoint | [NOW.md](NOW.md) | affected gate and experiment record |
| Start or interpret an experiment | [experiments/README.md](experiments/README.md) | affected gate, operation and immutable config under `../configs/` |
| Isaac training or checkpoint monitoring | [G05](gates/G05_isaac_training_first_loop.md) | [run_training.md](operations/run_training.md), [run_on_runpod.md](operations/run_on_runpod.md) |
| Isaac-to-MuJoCo / BankExam / Gate3 evidence | [G06](gates/G06_isaac_to_mujoco.md) | [policy contract](interfaces/policy_observation_action.md), relevant evaluation operation |
| Planner or planner-policy pairing | [G06](gates/G06_isaac_to_mujoco.md) | [run_planner.md](operations/run_planner.md), [ROS topics](interfaces/ros_topics.md), [build_and_test.md](operations/build_and_test.md) |
| Vendor C++ first tick or demo | [G06](gates/G06_isaac_to_mujoco.md) | [first-tick harness](operations/run_gate3_first_tick_harness.md), [end-to-end run](operations/run_pingpong_end_to_end.md) |
| New motion, retarget, TOPP or 2-vs-4 | [G05](gates/G05_isaac_training_first_loop.md) and [G08](gates/G08_blind_spot_improvements.md) | [motion pipeline](motion_pipeline.md), [counterfactual screen](operations/run_motion_gmr_counterfactual_screen.md), [spatial retarget](operations/run_motion_spatial_retarget_screen.md) |
| Continuous rally, recovery or random arrival | [G08](gates/G08_blind_spot_improvements.md) | [T1 contract](interfaces/t1_event_training_contract.md), [recovery prereg](operations/run_phase1_recovery_tuple_prereg.md) |
| Frames, racket face, observation or action semantics | affected file in [interfaces/](interfaces/) | relevant gate and operation |
| Restore ignored/local assets or environments | [setup_local_sync.md](operations/setup_local_sync.md) | [setup_environments.md](operations/setup_environments.md), [ASSET_POLICY.md](ASSET_POLICY.md) |
| Real robot preparation | [G01](gates/G01_real_preparation.md) and [G07](gates/G07_mujoco_to_real.md) | [run_deploy_dryrun.md](operations/run_deploy_dryrun.md); never infer hardware authorization from another gate |

## Gates

- [G00 — materials and harness](gates/G00_materials_and_harness.md)
- [G01 — real preparation](gates/G01_real_preparation.md)
- [G02 — data acquisition](gates/G02_data_acquisition.md)
- [G03 — data processing and physics calibration](gates/G03_data_processing_and_physics_calibration.md)
- [G04 — MuJoCo and Isaac modeling](gates/G04_sim_modeling_mujoco_isaac.md)
- [G05 — Isaac training first loop](gates/G05_isaac_training_first_loop.md)
- [G06 — Isaac-to-MuJoCo parity and vendor gate](gates/G06_isaac_to_mujoco.md)
- [G07 — MuJoCo-to-real deployment](gates/G07_mujoco_to_real.md)
- [G08 — blind-spot improvements](gates/G08_blind_spot_improvements.md)

When adding a new document category or changing a folder's role, update this file and
[PROJECT_MAP.md](PROJECT_MAP.md) in the same branch.
