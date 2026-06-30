# `hope_wbc_runner` — staged, safety-gated WBC controller for `model_15200`

Runs the validated `model_15200` ONNX from `/racket/command` (e.g. from
`planner_imitate`), builds the exact 180-D observation, infers **deterministically**
(mean action, no dither), logs the joint targets, and — only when explicitly
enabled — publishes low-level joint commands. Bring-up companion to
`planner_imitate`. **Does not drive hardware in dry-run/shadow.**

## Modes (param `mode`) and the publish gate

| mode | state | publishes joint cmds? |
|---|---|---|
| **`dry_run`** (default) | synthetic (perfect tracking of the reference) | **never** |
| **`shadow`** | real robot/sim (joints + IMU) | **never** (robot driven by something else) |
| **`hardware`** | real robot/sim | **only if** `hardware_enable` **and** `!estop` |

**Publish gate (defense in depth):** `mode == "hardware"` **AND** `hardware_enable`
**AND** `/hope/estop` is false. `dry_run`/`shadow` can never publish.

## Milestone 1 — dry-run, standalone (no robot/sim)

```bash
# Terminal A — fake planner publishes /racket/command (level 2 = forehand):
ros2 launch hope_planner planner_imitate.launch.py dry_run:=false level:=2

# Terminal B — WBC runner, dry-run: builds obs, runs model_15200, LOGS joint targets, no hardware:
ros2 launch hope_wbc_runner wbc_runner.launch.py \
    onnx_path:=$PWD/.../exported/policy.onnx mode:=dry_run csv_path:=/tmp/wbc_runner.csv
```
Expect: `[log-only/dry_run] forehand ts=34 phase=0.36 tts=... |act|=...` lines and a CSV
of per-tick action / `target_q`. **No joint commands are sent.**

> `planner_imitate dry_run:=false` only makes it *publish* `/racket/command`; the
> runner's `mode:=dry_run` is what guarantees no hardware output.

## Later milestones

* **Shadow** (predict against a running sim/robot, still no publish):
  ```bash
  ros2 launch hope_wbc_runner wbc_runner.launch.py onnx_path:=... mode:=shadow state_source:=ros
  ```
* **Hardware** (publish, gated):
  ```bash
  ros2 launch hope_wbc_runner wbc_runner.launch.py onnx_path:=... mode:=hardware state_source:=ros
  ros2 topic pub -1 /hope/hardware_enable std_msgs/Bool "{data: true}"   # arm
  ros2 topic pub -1 /hope/estop std_msgs/Bool "{data: true}"            # disable instantly
  ```

## Interface contracts (verified from the repo / the validated front-end)

* **Input:** `/racket/command` (`hope_msgs/RacketCommand`). Swing type from the target
  **Y sign** (−y forehand, +y backhand); timing from `time_to_strike`.
* **Obs (180-D)** in the verified training order (see `obs_builder.py`). Joints, gains,
  defaults, and the reference motion all come from the **ONNX metadata + side-outputs**
  (no external motion files needed). Joint order = ONNX `joint_names` (Isaac articulation order).
* **ONNX:** `obs[1,180]` + `time_step[1,1]` → `actions[1,31]` (mean) + reference side-outputs.
  Action decode: `target_q = default_q + action * action_scale`.
* **`learned_std.npy`:** NOT used for deterministic inference (`dither_scale: 0`). Only loaded
  if `dither_scale > 0` (off by default; dither is unnecessary and hurts the backhand).
* **Reference clock:** driven by the planner's `time_to_strike` so the clip's strike frame
  (forehand 0.36, backhand 0.50) lands at `tts = 0`. `seg_len = [95, 105]`.
* **Output (hardware only):** `joint_msgs/JointCommand` (`header + Command[]`; per joint
  `position` = `target_q`, `stiffness`/`damping` = ONNX `kp`/`kd`) — the position+PD interface
  the A3 backend expects. `joint_msgs` is **lazy-imported** (only hardware mode needs it built).
* **Estop:** `/hope/estop` (`std_msgs/Bool`). **Enable:** `/hope/hardware_enable` or the param.

## Frames

`/racket/command` from `planner_imitate` is `frame_id="base_link"` = the policy's
robot-ground / env-origin frame (+x fwd, +y left, +z above floor). The runner treats
those targets as the policy world frame directly. `frame_id="world"` is **not**
transformed yet (needs the unmeasured robot world pose) — the runner warns and proceeds.

## Open items (known gaps — read before hardware)

1. **Base/torso world pose (localisation).** The obs anchor-position term needs the robot
   torso *world* pose. In `dry_run` it's synthetic; in `ros` mode it falls back to a nominal
   upright pose (`nominal_base_height`/`torso_offset_z`) with a warning. **Wire a real base-pose
   estimate before trusting the anchor-position term on hardware.**
2. **`joint_msgs` build.** Only needed in hardware mode; ensure the `joint_msgs` package (from
   `agi/A3_MuJoCo_Sim/.../protocols/joint_msgs`) is on the ROS 2 path before `mode:=hardware`.
3. **State/command topic names** (`joint_state_topic`, `imu_topic`, `joint_command_topic`) are
   placeholders — map them to your A3 ROS 2 bridge.
4. **`onnxruntime`** must be installed in the node's Python (pip, not rosdep).

## Safe first-hardware-test checklist

1. [ ] `mode:=dry_run` first; confirm `/wbc_runner/diagnostics` + CSV look sane.
2. [ ] `ros2 topic echo /racket/command` shows expected `base_link` targets.
3. [ ] `mode:=shadow state_source:=ros` against the sim; compare logged `target_q` to MuJoCo.
4. [ ] Verify estop: `/hope/estop true` → diagnostics show `publishing=False`.
5. [ ] Only then `mode:=hardware` with `hardware_enable:=false`; arm via `/hope/hardware_enable`.
6. [ ] Keep the planner at `level:=0/1` (stand / slow forehand) for the first armed test.
7. [ ] Deterministic (`dither_scale:=0`). Watch action / `target_q` magnitudes before arming.
