# Policy Observation And Action

Status: Draft

## HITTER-Compatible Contract

The baseline policy should be compatible with the HITTER-style separation:

- Planner provides target racket position, target racket velocity, target racket normal, and time to strike.
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
- `swing_type` (1) — forehand `+1`, backhand `-1` for the unified policy

Current HITTER-aligned policy default: `swing_type` is included because the default task trains one
unified forehand+backhand policy. Desired racket normal is reward/critic-only, not an actor
observation.

Notes / corrections:

- Desired racket normal is not an actor observation in the current HOPE task. Actual racket normal is
  still privileged and simulation-only.
- "base forward vector" is NOT implemented; `projected_gravity` is the
  orientation cue.
- `HOPEPolicyCfg` removes inherited actor `base_lin_vel`; base linear velocity is kept critic-only for
  deployment alignment.

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

## Table-Tennis Physics Scene Observation (experimental)

`HOPE-TableTennis-AgibotA3-v0` is a G04/G08 physics/visualization scene, not the accepted deployment
WBC policy. Its current actor observation is robot proprioception plus ball state in the base frame:

- `base_lin_vel`
- `base_ang_vel`
- `projected_gravity`
- `joint_pos_rel`
- `joint_vel_rel`
- `last_action`
- `ball_pos_b`
- `ball_vel_b`

The critic mirrors these terms without observation noise. The action is A3 joint-position control with
the same `AGIBOT_A3_ACTION_SCALE` rule used by the tracking task. If this scene becomes a returner
training baseline, record dimensions, normalization, reward targets, and deploy compatibility here
before treating it as G05/G07 relevant.

## Contract Knobs

- Control rate: intended 50 Hz (decimation 4 over 200 Hz sim) — confirm from
  `base/sim_base.yaml` if unverified.
- Training target source: `HOPEPingPong` defaults to
  `racket.target_mode: uniform`, fixed `pos_x_range: [0.40, 0.40]`, sampled target `(y, z)` and
  racket velocity vector, and per-clip strike timing via `strike_phase_per_clip`
  (`[0.47, 0.33]` for the unified forehand+backhand clips).
- `racket.target_mode: reference_perturbed` remains a non-default option (it was
  the default on the pre-merge `rsi-on-wrap-progress-fix` branch): the target
  center is computed from each imitated clip's own strike-frame racket FK state
  (position, velocity, face normal), and the long-run distribution widens
  through success-gated perturbations. Either mode changes target generation
  only, not the observation/action tensor contract.
- Clip wrap for HOPE ping-pong does not teleport the robot mid-episode: the
  target and reference clip/time resample, but the policy must physically carry
  the body between swings (`motion.rsi_on_wrap: false` in the HOPE task YAMLs;
  `MotionCommandCfg.wrap_teleport` also defaults to false). True episode resets
  still use reference-state initialization, except for the `stand_start_prob`
  fraction of envs that start from the default stand pose.
- Normalization: per-term Unoise with `enable_corruption` on the policy
  observation group (e.g. `projected_gravity` Unoise +/-0.05,
  `racket_target_pos_b` Unoise +/-0.02).

## Update Rule

Any observation, action, scaling, normalization, latency, command frequency, or joint-order change must update this file and the relevant gate docs.
