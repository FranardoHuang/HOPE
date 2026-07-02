# Policy Observation And Action

Status: Partial

## Source Of Truth

The HOPE ping-pong policy contract is defined in:

- `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py`
- `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/actor_observation_contract.py`
- exported ONNX metadata written by `whole_body_tracking/utils/exporter.py`

Training and export now validate the actor observation layout against an explicit
contract before running.

## Recommended Actor Contract

The default HOPE training/export path is now the **deploy-parity** actor
contract:

- Task YAML default: `task=HOPEPingPongDeployParity`
- Gym task: `HOPE-PingPong-DeployParity-AgibotA3-v0`
- Actor input `obs`: shape `[1, 175]`
- Extra input `time_step`: shape `[1, 1]`
- Action output: shape `[1, 31]`

This contract removes every actor term that depends on the robot's true world
base position, because the current A3 deploy path does not have an accepted
world localizer for those quantities.

### Deploy-Parity 175-D Actor Layout

| Term | Dim | Deploy-side source |
| --- | ---: | --- |
| `command` | 62 | baked reference clip joint pos/vel |
| `motion_anchor_ori_b` | 6 | IMU orientation + reference clip |
| `base_ang_vel` | 3 | pelvis IMU gyro |
| `joint_pos` | 31 | joint encoders, offset from default |
| `joint_vel` | 31 | joint encoders |
| `actions` | 31 | previous policy action |
| `projected_gravity` | 3 | pelvis IMU |
| `racket_target_pos_b` | 3 | planner target minus current racket FK, in yaw-heading base frame |
| `racket_target_vel_w` | 3 | planner desired racket velocity |
| `time_to_strike` | 1 | strike clock / reference clock |
| `swing_type` | 1 | forehand `+1` / backhand `-1` |

Notes:

- `racket_target_pos_b` is **reframed** in deploy-parity mode. It is no longer
  `target_w - base_pos_w`; it is `target_w - racket_fk_w`, rotated into the
  yaw-heading frame, so world base position cancels out.
- `base_lin_vel` is not part of the actor. It remains critic-only.
- Critic observations may stay privileged in simulation; only the actor must be
  deploy-honest.

### Removed Versus Legacy Full Actor

| Term | Dim | Why removed from actor |
| --- | ---: | --- |
| `motion_anchor_pos_b` | 3 | needs true world base/torso position |
| `base_target_pos_b` | 2 | needs true world base position |

## Legacy Full Actor Contract

The old comparison path is still available for sim-only ablations:

- Task YAML: `task=HOPEPingPong`
- Gym task: `HOPE-PingPong-AgibotA3-v0`
- Actor input `obs`: shape `[1, 180]`

The legacy full actor adds the two removed base-position terms above and keeps
`racket_target_pos_b` in its old base-position-dependent form. Do not treat this
path as deploy-safe.

## Critic Observation

The critic is privileged. On top of the actor terms it may include actual racket
state from forward kinematics and other simulation-only information, including:

- desired racket normal
- actual racket position / velocity / normal
- episode time left

This is acceptable because the critic is not deployed to hardware.

## Action Contract

- Action term: joint position targets with `use_default_offset=True`
- Action dimension: `31`
- Decode: `q_des = default_joint_pos + action * action_scale`
- Joint order, default positions, and action scale must match the exported ONNX
  metadata for the exact artifact under test

## Current Deploy Gap

The **training/export** default is now the 175-D deploy-parity actor contract.
However, the currently observed local A3 ping-pong deploy runner is still the
older **180-D** front-end that uses `perfect_tracking` / reference-filled world
position placeholders. That runner is not yet an accepted deploy-parity front
end.

Before claiming Isaac-to-deploy observation parity on hardware, the deploy-side
obs builder must be ported and verified against the 175-D deploy-parity layout
and exported ONNX metadata.

## Update Rule

Any observation, action, scaling, normalization, latency, command frequency,
joint-order, or deploy/localizer assumption change must update this file and the
relevant gate docs.
