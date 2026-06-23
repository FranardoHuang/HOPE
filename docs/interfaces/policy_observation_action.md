# Policy Observation And Action

Status: Draft

## HITTER-Compatible Contract

The baseline policy should be compatible with the HITTER-style separation:

- Planner provides target racket position, target racket velocity, and time to strike.
- WBC policy combines planner target, robot state, and previous action.
- Policy outputs desired joint positions.

## Actor Observation (implemented)

Source of truth: `hope_env_cfg.py`. The actor (policy) group is the BeyondMimic
proprioceptive + motion terms inherited from `tracking_env_cfg.py` `PolicyCfg`,
PLUS the following appended HOPE terms:

- `projected_gravity` (3) — Unoise +/-0.05; this is the orientation cue
- `base_target_pos_b` (2)
- `racket_target_pos_b` (3) — Unoise +/-0.02
- `racket_target_vel_w` (3)
- `time_to_strike` (1)

`swing_type` is omitted by default (it is constant per forehand/backhand policy).

Notes / corrections:

- "base forward vector" is NOT implemented; `projected_gravity` is the
  orientation cue.
- The inherited BeyondMimic proprio terms include `base_lin_vel` (3), which
  sits in the ACTOR group. Base linear velocity is not directly observable on
  hardware, so this is a real sim-to-real concern (not a typo).

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
- Dimension = 31 (training) / 29 (deploy). The 2 neck joints
  (`head_yaw_joint`, `head_pitch_joint`) are excluded from the deploy policy and
  driven at deploy `kp=40`/`kd=2` via `ExpandToBackend`. See
  [joint_order_and_robot_state.md](joint_order_and_robot_state.md).
- Decoder target = `action * action_scale + default_angle`, with per-joint
  `action_scale = 0.25 * effort_limit / stiffness`. This must match the deploy
  action decoder (`a3_action_scale`).

## Contract Knobs

- Control rate: intended 50 Hz (decimation 4 over 200 Hz sim) — confirm from
  `base/sim_base.yaml` if unverified.
- Normalization: per-term Unoise with `enable_corruption` on the policy
  observation group (e.g. `projected_gravity` Unoise +/-0.05,
  `racket_target_pos_b` Unoise +/-0.02).

## Update Rule

Any observation, action, scaling, normalization, latency, command frequency, or joint-order change must update this file and the relevant gate docs.
