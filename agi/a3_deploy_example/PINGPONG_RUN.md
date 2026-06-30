# model_15200 ping-pong runner — run & validation guide

Separate native runner `a3_deploy_onnx_ref_pingpong` (AGI's `a3_deploy_onnx_ref`
is untouched). Front-end = our verified C++ port of model_15200 (180-obs/31-act,
neck PASSIVE). Reuses AGI's A3AimrtBackend + A3PolicyDriver + watchdog + safe-halt.

Binary: `dist/a3_deploy_x86_64/a3_deploy_onnx_ref_pingpong`
Config: `dist/a3_deploy_x86_64/config/a3_runtime_config.pingpong.yaml` (→ `../models/model_15200.onnx`, iceoryx)

## Modes (keyboard at runtime, or `--start MODE`)
`p`=PASSIVE (limp, zero gain) · `s`=PD_STAND (hold nominal, modest PD) ·
`h`=SHADOW (compute, **no publish**) · `m`=MOTION (publish) ·
`0`/`1`=swing level (0 hold-windup, 1 forehand) · `[`/`]`=gain_scale −/＋ · `q`=quit.
Flags: `--start`, `--level`, `--gain-scale F`, `--stand-kp`, `--stand-kd`.

## A. MuJoCo standalone validation (do BEFORE hardware)

Transport must match: runner uses `a3_aimrt_config.iceoryx.yaml`; pick a sim cfg
whose `/body_drive/*_joint_{state,command}` are on **iceoryx** (a3_t2d5 is our
robot variant; if the runner shows `rate=0`/safe-halt, the joint topics are on
ros2 → switch the runner to `a3_aimrt_config.ros2.yaml`, or use an iceoryx sim cfg).

**Terminal 1 — sim** (starts iox-roudi + MuJoCo; GUI):
```bash
cd agi/a3_deploy_example/mujoco_sim_standalone
./run.sh a3_t2d5_cfg.yaml
```

**Terminal 2 — runner**, staged exactly like your plan:
```bash
cd agi/a3_deploy_example/dist/a3_deploy_x86_64
RT=config/a3_runtime_config.pingpong.yaml
# 1. probe / shadow first: compute, NO publish (watch rate, |act|, ts sweep)
./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT --start shadow
#    -> expect rate≈50Hz, ts sweeping the clip, |act| bounded, robot still
# 2. dry-run stand check: PD_STAND (press 's') -> robot holds nominal
# 3. full closed-loop: press 'h'->'m' (MOTION), level 0 first (press '0'),
#    then level 1 (press '1') for the forehand. Watch neck/wrist stability.
```

## B. Compare against Python wbc_runner / sim2sim
- **stand-hold:** PD_STAND pose ≈ nominal (hip −0.131, knee 0.247, …).
- **q_des continuity:** no jumps tick-to-tick (status `|act|` smooth).
- **neck stability:** head must NOT buzz (neck is passive: q=0, kp=40, kd=2).
- **wrist stability:** wrists steady (low-inertia; if buzz, lower `--gain-scale`).
- **recognizable forehand:** level 1 produces a forehand swing matching the clip.

## C. Cross-build for rockchip (only after A passes)
```bash
cd agi/a3_deploy_example
bash scripts/build_a3_deploy_pkg.sh --arch rockchip   # builds a3_deploy_onnx_ref_pingpong too
# stage model_15200.onnx into dist/a3_deploy_rockchip/models/ + a runtime cfg
```
NOTE: this laptop was missing the zmq dev headers; vendored to
`thirdparty/zmq_shim/include/` (zmq.h v4.3.5 + cppzmq zmq.hpp). If the rockchip
toolchain also lacks them, point its `ZMQ_INCLUDE_DIR/CPPZMQ_INCLUDE_DIR` there.

## D. Hardware — HOISTED only, staged (your sequence)
PASSIVE → PD_STAND → SHADOW(no-command) → low-gain MOTION (`--gain-scale 0.4`)
→ level 0 → level 1. Physical e-stop in hand. Neck passive. Scripted targets
only (no live planner yet).
