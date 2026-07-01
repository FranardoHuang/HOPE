# Policy Observation And Action

Status: Draft

## HITTER-Compatible Contract

The baseline policy should be compatible with the HITTER-style separation:

- Planner provides target racket position, target racket velocity, and time to strike.
- WBC policy combines planner target, robot state, and previous action.
- Policy outputs desired joint positions.

## Current `model_15200.onnx` Contract

Source of truth for the current exported ping-pong deployment artifact is the
training/eval path around `HOPEPingPong`, `hope_env_cfg.py`, and
`mujoco_eval_onnx.py`, plus ONNX metadata once inspected in the target runtime.

The observed deploy contract for `model_15200.onnx` is:

- Input `obs`: shape `[1, 180]`.
- Input `time_step`: shape `[1, 1]`.
- Action output: shape `[1, 31]`.
- Additional reference outputs may be present and should be logged, not used as
  hardware commands.

The 180-D actor observation order is:

| Slice | Dimension | Meaning |
| --- | ---: | --- |
| `command` | 62 | Reference joint positions and velocities |
| `motion_anchor_pos_b` | 3 | Motion anchor position in body/base frame |
| `motion_anchor_ori_b` | 6 | Motion anchor orientation representation |
| `base_ang_vel` | 3 | Base angular velocity |
| `joint_pos` | 31 | Joint position offset from default |
| `joint_vel` | 31 | Joint velocity |
| `last_action` | 31 | Previous policy action |
| `projected_gravity` | 3 | Orientation cue |
| `base_target_pos_b` | 2 | Target base position in body/base frame |
| `racket_target_pos_b` | 3 | Target racket position in body/base frame |
| `racket_target_vel_w` | 3 | Target racket velocity in world frame |
| `time_to_strike` | 1 | Time-to-strike scalar |
| `swing_type` | 1 | Forehand/backhand or unified-policy swing flag |

Notes / corrections:

- `swing_type` is present in the current unified policy contract.
- Base linear velocity is not part of the current 180-D actor observation.
- "base forward vector" is not implemented; `projected_gravity` is the
  orientation cue.
- Current hardware default is `loc_mode=perfect_tracking`: base/torso world
  position is filled from the reference trajectory, while IMU orientation
  remains real. This is a deploy placeholder, not a true world localizer.
- `oracle` pose feeding is simulation-only today. The current
  `a3_deploy_onnx_ref_pingpong` runner does not yet subscribe directly to HOPE
  mocap/VRPN topics on hardware.
- Hardware deployment must verify how base pose, torso pose, projected gravity,
  and anchor terms are estimated before accepting real motion quality.
- `level 0` vs `level 1` is part of the current runtime contract: `level 0` is
  a ready/windup hold state, while `level 1` releases the full-body swing.
  A good hoist swing at `level 1` does not by itself prove ground support.

## Critic-Only (privileged) Observation (implemented)

The critic group is privileged (never available on hardware). On top of the
actor terms it adds the actual racket state computed via FK:

- `base_target_pos_b`
- `racket_target_pos_b`
- `racket_target_vel_w`
- `racket_target_normal_w`
- `time_to_strike`
- `racket_pos_b` (actual, via FK)
- `racket_lin_vel_w` (actual, via FK)
- `racket_normal_w` (actual, via FK)
- `episode_time_left`

## Action

- `ActionsCfg.joint_pos` with `joint_names=[.*]`, `use_default_offset=True`.
- ACTION = desired joint position targets.
- Dimension = 31 for the current HOPE `model_15200.onnx` artifact.
- The official AGI deploy policy uses a different 29-DOF action contract and is
  not a drop-in front-end for this model.
- When scattering to the 31-slot SDK/backend command layout, head and neck
  handling must be explicit. The AGI baseline pins or holds those slots rather
  than treating them as policy-controlled HITTER joints.
- Decoder target = `action * action_scale + default_angle`, with the action
  scale and default angle matching the training/export contract for the exact
  model artifact. Record the model fingerprint in the relevant gate before
  hardware use.

## Contract Knobs

- Control rate: intended 50 Hz (decimation 4 over 200 Hz sim) — confirm from
  `base/sim_base.yaml` if unverified.
- Training target source: `HOPEPingPong` defaults to
  `racket.target_mode: reference_perturbed`, meaning the command term samples
  the desired racket pos/vel/normal around the reference motion's strike-frame
  racket state. This changes target generation, not the observation/action
  tensor contract.
- Normalization: per-term Unoise with `enable_corruption` on the policy
  observation group (e.g. `projected_gravity` Unoise +/-0.05,
  `racket_target_pos_b` Unoise +/-0.02).

## Update Rule

Any observation, action, scaling, normalization, latency, command frequency, or joint-order change must update this file and the relevant gate docs.
