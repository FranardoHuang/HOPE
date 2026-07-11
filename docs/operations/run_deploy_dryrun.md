# Run Deploy Dry-Run

Status: Draft

## Current Materials

- Active tracked ping-pong deploy tree: `agi/a3_deploy_example/` — the C++ runner `a3_deploy_onnx_ref_pingpong` (used for the first successful sim-to-real, 2026-07-02), build script `scripts/build_a3_deploy_pkg.sh`, and runbooks `PINGPONG_RUN.md`, `PINGPONG_DEPLOY_ALIGNMENT.md`, `HARDWARE_BRINGUP_CHECKLIST.md`, `MUJOCO_VALIDATION_RUNBOOK.md`
- Older vendor reference subset: `agi/code_deployment/a3_deploy_example`
- Full local payload: `vendor_assets/agibot/a3_deploy_example_full`
- Deployment docs: `agi/code_deployment/A3 deploy example.md`
- Backend guide: `agi/code_deployment/RobotIOBackend 架构与策略适配指南.md`

## Task Setup

Use the target Agibot deploy/MuJoCo environment, not the ROS planner environment and not the Isaac training environment.

This task does require ignored local assets. Before running deploy or standalone MuJoCo dry-runs, restore:

```text
vendor_assets/agibot/a3_deploy_example_full/
```

Check:

```bash
test -d vendor_assets/agibot/a3_deploy_example_full && echo "Agibot full deploy payload present"
```

The active tracked ping-pong deploy tree is `agi/a3_deploy_example/`; the older vendor reference subset in `agi/code_deployment/a3_deploy_example/` remains useful for code review and integration planning. The full ignored payload is needed for runtime assets such as models, sysroots, prebuilt libraries, and standalone runtime files.

## Plain-MuJoCo PD_STAND diagnostic

Before involving AimRT, a planner or a policy, the tracked vendor MJCF can be inspected with a
small diagnostic. First bind the exact source bytes without importing MuJoCo:

```bash
python3 scripts/view_a3_stand.py --identity-only
```

At the 2026-07-12 selective-port audit the tracked identities were vendor
`a3_pingpong.xml=2ab1cd31...3feb97` and production parameter header
`df73e3f6...c5c8d8`. The script parses `a3_default_angles`, `a3_pd_stand_kps` and
`a3_pd_stand_kds` directly from that header; it does not maintain a second gain table. The header's
contract is 29 DOFs, so `head_yaw_joint` and `head_pitch_joint` remain passive. Do not describe
ad-hoc head gains as production PD_STAND.

With the MuJoCo Python binding installed, run the numerical diagnostic:

```bash
python3 scripts/view_a3_stand.py \
  --check --duration 10 \
  --report-json /tmp/a3_stand_diagnostic.json
```

It resets the vendor `stand` keyframe, then holds the production default pose with production
PD_STAND gains. The JSON binds the two root-source SHAs and reports actual integrator/timestep,
finite qpos/qvel/qacc/ctrl/actuator-force arrays, pelvis-z range/drift, maximum pelvis tilt, and
left/right/both-foot floor-contact fractions. It does not hash the MJCF's transitive mesh closure;
a formal Gate3 runtime still needs the full resolved asset closure.
Defaults flag z drift over `0.15 m`, pelvis tilt over `20 deg`, or both-foot contact under `0.90`.
These are diagnostic tripwires, not frozen Gate3 acceptance thresholds. To inspect a frame:

```bash
python3 scripts/view_a3_stand.py --snapshot /tmp/a3_stand.png
```

The script does not alter the vendor MJCF or choose a different integrator. Because the tracked
MJCF omits an integrator attribute, the tool reports MuJoCo's realized default rather than silently
"fixing" it. A pass proves only this plain-MuJoCo PD stand diagnostic. It does not run the planner,
policy, AimRT/backend, first tick, continuous Gate3/Gate3B paper or hardware. On the 2026-07-12 Mac
audit, `--identity-only`, pycompile and four host tests passed; the 10-second run remained `not_run`
because the MuJoCo Python binding was absent.

## Ping-Pong Dry-Run

For the ping-pong runner, the actual dry-run sequence starts from the package built by `agi/a3_deploy_example/scripts/build_a3_deploy_pkg.sh`; from the build output directory:

```bash
./run_a3_pingpong.sh --dry-run
```

then the inference/latency probe, shadow mode, and the staged bringup PASSIVE -> PD_STAND -> SHADOW -> MOTION. Follow [../../agi/a3_deploy_example/PINGPONG_RUN.md](../../agi/a3_deploy_example/PINGPONG_RUN.md) and [../../agi/a3_deploy_example/PINGPONG_DEPLOY_ALIGNMENT.md](../../agi/a3_deploy_example/PINGPONG_DEPLOY_ALIGNMENT.md) §0.

## Safety Order

Do not jump directly to hardware motion.

1. Build deploy code in the target environment.
2. Start backend/sync without loading policy and without publishing commands.
3. Run inference latency probe without command output.
4. Verify joint order and command scaling.
5. Verify safe halt.
6. Only then plan low-gain bounded hardware command tests.

## Required Documentation Before Hardware

- [../interfaces/joint_order_and_robot_state.md](../interfaces/joint_order_and_robot_state.md) must be filled.
- G07 must list exact build command, dry-run command, latency result, and safe halt result.
