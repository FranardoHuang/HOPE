# Policy Observation And Action

Status: Implemented for 110/175/177/180; 179/181 training/evaluation implemented but deploy wire
is intentionally blocked pending the normal/station contract day (audited 2026-07-10).

## HITTER-Compatible Contract

The baseline policy keeps the HITTER-style separation:

- Planner provides target racket position, target racket velocity, target racket normal, and time to strike.
- WBC policy combines planner target, robot state, and previous action.
- Policy outputs desired joint positions.

## Actor Observation (implemented): 175-D deploy-parity

Source of truth: `HOPEPingPongDeployParityAgibotA3EnvCfg` in `hope_env_cfg.py`
(`HOPEPingPongRealSensor` is a backward-compat alias). The canonical layout with per-term hardware
sources is codified and asserted in
`hope_training/whole_body_tracking/scripts/realsensor_obs_reference.py` and checked by
`scripts/verify_realsensor.py`; the C++ deploy runner builds the identical layout
(`pp_obs_builder.hpp`) and auto-detects 175 vs 180 from the ONNX input dim.

| # | Term | Dim | On-hardware source |
| --- | --- | --- | --- |
| 0 | `command` | 62 | reference clip future joint pos/vel (baked into the ONNX) |
| 1 | `motion_anchor_ori_b` | 6 | relative orientation ref-vs-robot (IMU + clip); no base position |
| 2 | `base_ang_vel` | 3 | pelvis IMU gyro |
| 3 | `joint_pos` | 31 | joint encoders (q − default_q) |
| 4 | `joint_vel` | 31 | joint encoders (dq) |
| 5 | `actions` | 31 | previous action |
| 6 | `projected_gravity` | 3 | pelvis IMU (gravity in base frame), Unoise ±0.05 |
| 7 | `racket_target_pos_b` | 3 | REFRAMED: `yaw(base)⁻¹ · (target_w − racket_FK_w)`, Unoise ±0.02 |
| 8 | `racket_target_vel_w` | 3 | planner desired racket velocity (world) |
| 9 | `time_to_strike` | 1 | swing clock (from clip strike phase) |
| 10 | `swing_type` | 1 | forehand `+1` / backhand `−1` (planner) |

Total = 175. Removed vs the legacy 180-D `full` layout:

- `motion_anchor_pos_b` (3) — reference torso position error; needs the world base pose.
- `base_target_pos_b` (2) — base-repositioning target; needs the world base pose.

The racket-target reframe makes the term base-position-free: because `quat_rotate_inverse` is linear
in its vector argument, `R⁻¹(target − racket) = (target rel base) − (racket rel base)` for ANY base
position — verified numerically in `realsensor_obs_reference.py`.

Notes:

- Desired racket normal is NOT an actor observation (HITTER Table I: normal is a reward target
  only). `base_lin_vel` is critic-only. `swing_type` is included because the default task trains
  one unified forehand+backhand policy.
- Rationale for base-position freedom: the mocap DOES stream the robot base pose during play
  (300 Hz, `/P1/pose` — see the deploy-available signal set below), but that VRPN link is not
  bridged into the deploy runner, and independence from it is a deliberate robustness choice. The
  earlier "no localizer on the real A3" wording was inaccurate.
- The legacy 180-D `full` contract (`task=HOPEPingPong`, model_15200 lineage) is kept for
  comparison only; it depends on world-base-position terms and is not deploy-honest. The deploy
  runner still accepts 180-D ONNX for legacy checkpoints.

### Other registered actor layouts

| Dim | Contract | Delta / source | C++ publish status |
| --- | --- | --- | --- |
| 177 | `hitter_footwork` | 175 layout with `base_target_pos_b(2)` inserted after projected gravity; requires fresh external/oracle base localization. | Supported, but publication fails closed without fresh localization. |
| 179 | `deploy_parity_face179` | Exact 175 prefix + tail `racket_target_normal_cmd(3), rho(1)`; demanded normal is the delayed atomic +Y/A-frame planner command. | Blocked: flat wire v1 carries no demanded normal/rho. |
| 181 | `deploy_parity_station181` | Exact 179 prefix + tail `station_anchor_err_b(2)`. | Blocked: wire and the unique station/normal term order are not frozen. |
| 110 | `hitter_pure` | HITTER Table-I style: 99-D proprio prefix + base forward(2), station delta(2), racket target rel base(3), target velocity(3), tts(1); no reference command or swing flag. | Supported; requires fresh localization and metadata-bound per-side station geometry. |

Do not infer a contract from width alone. Formal consumers require the registered name, mode,
ordered term names/dims and total dimension to agree. In particular, merely accepting 179/181 in
the C++ input-shape whitelist would turn a right-width/wrong-columns model into a hardware command;
the runner intentionally rejects them until flat-wire v2 and the 181 order are frozen together.

## Critic (privileged) Observation (implemented)

The critic group is unchanged between full and deploy-parity (~318-D) and is never deployed. Base
`PrivilegedCfg` terms: `command` 62, `motion_anchor_pos_b` 3, `motion_anchor_ori_b` 6, `body_pos`
42 (14 tracked bodies × 3), `body_ori` 84 (14 × 6), `base_lin_vel` 3, `base_ang_vel` 3,
`joint_pos` 31, `joint_vel` 31, `actions` 31 — all noise-free. HOPE additions: `base_target_pos_b`
2, `racket_target_pos_b` 3, `racket_target_vel_w` 3, `racket_target_normal_w` 3, `time_to_strike`
1, plus sim-only actual racket FK state `racket_pos_b` 3, `racket_lin_vel_w` 3, `racket_normal_w`
3, and `episode_time_left` 1. Privileged sim-only info is allowed here by the training contract.

## Action

- `ActionsCfg.joint_pos` with `joint_names=[.*]`, `use_default_offset=True`.
- Dimension = **31** in training AND in the deployed ping-pong runner: the ONNX outputs 31 actions
  (incl. `head_yaw_joint`/`head_pitch_joint`); the runner scatters them to the 31 SDK slots
  (`MakeA3Layout31`) and overrides the neck slots post-decode to `q=0, kp=40, kd=2`. The 29-DOF
  `ExpandToBackend` view belongs to AGI's official reference runner, not the HOPE ping-pong path.
  See [joint_order_and_robot_state.md](joint_order_and_robot_state.md).
- Decoder target = `action * action_scale + default_angle`, per-joint
  `action_scale = 0.25 * effort_limit / stiffness`. The policy target first reproduces the
  schema-3 training soft q-des limits in actor order; a separate outer SDK-order hard clamp protects
  stand/reference/blend and future command sources. The runner publishes `{q_des, kp, kd}` to the
  implicit-PD backend (`dq_des = tau_ff = 0`) only while its requested/measured effort envelope and
  authorization generation remain valid.

## ONNX Metadata Contract

The whole obs/action contract travels with the model. `scripts/play.py` bakes into the ONNX:
`joint_names`, `default_joint_pos`, `action_scale`, `joint_stiffness` (kp), `joint_damping` (kd),
`body_names`, and the clip clock keys `clip_seg_lengths` / `clip_strike_phases`. Consumers
hard-validate at load and fail on any missing key, non-bijective joint map, obs input other than
[1,175]/[1,180], or any kp/kd ≤ 0 (zero-gain guard): the C++ runner (`pp_onnx_policy.hpp`), the
MuJoCo evaluator (`mujoco_eval_onnx.py`), and the contract diff tool
(`scripts/inspect_a3_deploy_contract.py`).

### Formal execution provenance (training-contract schema 3)

`hope_metadata_schema_version=2` remains the ONNX packaging/layout schema. A distinct immutable
`training_contract_schema_version=3` now binds the checkpoint and export to execution facts read
from the instantiated environment, not copied from YAML comments:

- articulation and action joint order (identity is required), default q, action scale and
  `use_default_offset`;
- per-joint actuator integration type, kp/kd, armature, effort/velocity
  limits, PhysX friction backend/semantics/units, the 31 soft q-des limit pairs
  and whether q-des clamp was active;
- physics dt, policy dt and decimation (`policy_dt = physics_dt * decimation`);
- exact actor term names/dims/history, articulation and tracked body
  order/indices, anchor and motion segment lengths, per-clip FPS and the
  schema-2 motion-kinematics contract;
- task timing/target/bank facts needed to prevent exporting an old actor under a new evaluation
  recipe.
- the canonical racket-point identity and wrist-local offset. See
  [racket_contact_geometry.md](racket_contact_geometry.md) for the distinction
  between the URDF site, physical face centre and ball centre at contact.

The adjacent `params/training_contract.json` is content-addressed; its SHA and schema are embedded
in every newly saved checkpoint. Only a fresh schema-3 run or a resume from an exact SHA-bound
schema-3 checkpoint can keep `training_contract_lineage_exact=1`. Legacy/missing/mismatch overrides
remain diagnostic forever and cannot be “washed” into formal status by one continuation save.
Native and standalone exporters verify checkpoint↔JSON binding, write
`source_checkpoint_sha256`, and derive normalization truth from the actual graph/checkpoint.

Publish-capable C++ and formal BankExam require: metadata schema 2, exact training schema 3,
baked empirical normalization, self-consistent dt/decimation, `qdes_clamp=1`, 31 soft-limit pairs,
and the source checkpoint/contract hashes. Older artifacts may only be opened under process-wide
no-publish diagnostics.

Formal evaluation additionally binds runtime facts that do not belong in the
training ONNX: immutable question schedule, common MJCF `stand` ready-state,
MJCF content and resolved MuJoCo execution-contract SHA. A teacher-reference
reset or direct PhysX-friction-number proxy forces
`evaluation_contract_exact=false`.

## Deploy-Available Signal Set

What the real system can observe (team contract, 2026-07):

- Robot-side (always): joint encoders (q, dq), pelvis IMU (quat, gyro), torso IMU (orientation),
  previous action.
- Mocap during PLAY (ChingMu/VRPN, 300 Hz): robot base (pelvis) pose, ball position. Ball
  rotation/spin is planned for the physics-modeling phase — not yet measured.
- Mocap during DATA-COLLECTION/physics-calibration only: additionally racket pose, table 4
  corners, net 2 corners.

Rules: actor observations must be built only from deploy-available signals; rewards run in sim
only and may use privileged state; the critic may be privileged. The current 175-D actor uses only
robot-side signals + planner targets (no mocap terms at all). NOTE: when the mocap→planner loop is
bridged, the HOPE-world → robot-frame target transform will need the mocap base pose at the
interface boundary even though the actor obs does not — that transform currently has no owner and
must be designed with the bridge (see G07 Next Steps).

## Table-Tennis Physics Scene Observation (experimental)

`HOPE-TableTennis-AgibotA3-v0` is a G04/G08 physics/visualization scene, not the accepted deployment
WBC policy. Its current actor observation is robot proprioception plus ball state in the base frame:
`base_lin_vel`, `base_ang_vel`, `projected_gravity`, `joint_pos_rel`, `joint_vel_rel`,
`last_action`, `ball_pos_b`, `ball_vel_b` (position/velocity only — no spin). The critic mirrors
these without noise. If this scene becomes a returner baseline, record dimensions, normalization,
reward targets, and deploy compatibility here first.

## Contract Knobs

- Control rate is model metadata, currently 50 Hz (physics dt 0.005 × decimation 4). Formal
  consumers reject a runtime rate that disagrees with the model instead of assuming 50 Hz.
- Training target source: `target_mode: uniform` with per-clip 3-D position and velocity boxes
  centered on the reference blade strike state (`pos_range_per_clip` / `vel_range_per_clip` in
  `cfg/task/HOPEPingPongDeployParity.yaml`); `strike_phase_per_clip: [0.47, 0.333]` on the
  re-grounded `_hopex` clips. The fixed `x=0.4` hit plane with (y,z)-only sampling is superseded.
- `racket.target_mode: reference_perturbed` remains a non-default option (it was
  the default on the pre-merge `rsi-on-wrap-progress-fix` branch): the target
  center is computed from each imitated clip's own strike-frame racket FK state
  (position, velocity, face normal), and the long-run distribution widens
  through success-gated perturbations. Either mode changes target generation
  only, not the observation/action tensor contract.
- Clip wrap for HOPE ping-pong does not teleport the robot mid-episode: the
  target and reference clip/time resample, but the policy must physically carry
  the body between swings (`MotionCommandCfg.wrap_teleport` defaults to false;
  the HOPE task YAMLs keep it explicit as `motion.wrap_teleport: false`). True episode resets
  still use reference-state initialization, except for the `stand_start_prob`
  fraction of envs that start from the default stand pose.
- Normalization: per-term Unoise with `enable_corruption` on the policy group (values in the actor
  table above).

## Update Rule

Any observation, action, scaling, normalization, latency, command frequency, or joint-order change must update this file and the relevant gate docs.
