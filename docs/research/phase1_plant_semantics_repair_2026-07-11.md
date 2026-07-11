# Phase-1 Plant Semantics Repair Contract (2026-07-11)

Status: **preregistered design, blocked on calibration evidence; no launch and no
real-robot authorization**.

Machine-readable contract:
[`configs/phase1_plant_semantics_repair_prereg_20260711.json`](../../configs/phase1_plant_semantics_repair_prereg_20260711.json).

Validator:
[`scripts/validate_phase1_plant_semantics_prereg.py`](../../scripts/validate_phase1_plant_semantics_prereg.py).

## Decision

The apparent tension is real, but the two conclusions answer different
questions:

- current `SZ` (shared face, all-zero joint friction) is the only implemented
  cell whose **friction setting** can be replayed with identical meaning in the
  current Isaac and MuJoCo evaluators. It may therefore produce a fresh,
  execution-protocol-exact Phase-1 score;
- `SZ` is not a deployment-plant claim. The 2026-07-07 frozen-policy probe
  recorded virtual hit `0.9997 -> 0.63` and fall `0.27 -> 0.87` when moving from
  the zero plant to the historical non-zero plant. The raw probe artifact is not
  content-addressed in this repository, so this is a directional deployment
  blocker, not calibration data;
- current `SP/LP` are not the missing controls. They numerically reused MuJoCo
  `frictionloss` values as PhysX coefficients even though the quantities have
  different units and laws. They can measure a historical configuration toggle,
  not a physical friction main effect;
- a deployment-qualified `SC` cell does not yet exist. The name is reserved for
  a fresh shared-face cell backed by measured friction, one latent physical model,
  two separately fitted engine adapters and a new versioned evaluation contract.

No current Phase-1 recipe, checkpoint, worker or Pod should be changed to make
this statement true. The repair is the next independent plant axis.

## What the two engines mean

| Runtime field | Physical meaning | Unit | Load dependence | Consequence |
| --- | --- | --- | --- | --- |
| Isaac Lab 2.1 `ImplicitActuatorCfg.friction` / legacy PhysX `setFrictionCoefficient` | Available resisting joint force/torque is bounded by a unitless coefficient times an approximation of the spatial force transmitted through the joint. The solver may hold the joint at rest when the available impulse is sufficient. | dimensionless | yes, through transmitted spatial force; also solver/step dependent | A number such as `1.1971` is not `1.1971 N m`. |
| MuJoCo joint `frictionloss` / `dof_frictionloss` | Load-independent dry-friction constraint: an upper bound on absolute generalized friction force. Damping is a separate velocity-linear passive term. | generalized force; for each A3 hinge joint, `N m` | no | It cannot reproduce a non-zero PhysX coefficient by copying the number. |

Primary sources:

- [Isaac Lab 2.1 actuator configuration](https://isaac-sim.github.io/IsaacLab/v2.1.0/_modules/isaaclab/actuators/actuator_cfg.html)
- [PhysX reduced-coordinate articulation joint friction](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/_api_build/class_px_articulation_joint_reduced_coordinate.html)
- [MuJoCo joint XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-joint)
- [MuJoCo friction-loss computation](https://mujoco.readthedocs.io/en/latest/computation.html#friction-loss)

There is no universal conversion `mu_physx -> tau_frictionloss`. PhysX's bound
depends on an internal transmitted spatial-force approximation, pose, load,
controller and solver. MuJoCo's bound is a constant generalized force. Equal
numbers are neither necessary nor sufficient for equal behavior. The one common
point already implemented is zero.

The repair therefore maps by **behavior**, not by parameter equality: fit a
content-addressed physical friction model, fit each engine adapter separately,
then compare their measured generalized resisting torque on the same frozen
load x velocity x direction probes.

## Keep the claim labels separate

| Label | What it proves | What it does not prove |
| --- | --- | --- |
| `training_lineage_exact` | Fresh checkpoint, schema-2 motion, bank and instantiated hard contract are SHA-bound. | Friction is physically calibrated. |
| `evaluation_protocol_exact` | Immutable paper, denominator, reset and execution recipe were replayed. | The two engines have identical dynamics. |
| `plant_adapter_replay_exact` | A versioned latent model and adapter were replayed byte-for-byte inside declared support. | The adapter matches A3, unless calibration also passed. |
| `cross_engine_plant_equivalence_passed` | Both adapters match the same physical probe and each other within frozen tolerances. | Hardware safety or table-tennis quality. |
| `deployment_plant_calibrated` | Accepted A3/vendor bench data, session-held-out fit and uncertainty gates passed. | Policy robustness on that plant. |
| `deployment_candidate` | All above, the paired training/eval matrix, and independent G07 safety gates passed. | Permission to bypass staged hardware bring-up. |

Thus a fresh `SZ` checkpoint may have exact lineage and an exact current
BankExam protocol while `deployment_plant_calibrated=false` and
`deployment_candidate=false`. That is not a contradiction.

## Repair stages

### P0: engine semantics probes (local/simulator only)

Before fitting anything, implement one immutable probe schedule for a scalar
hinge and the full A3 articulation:

1. freeze engine version, solver/integrator, physics step, asset SHA, joint
   order, pose, commanded drive, external load, direction and initial velocity;
2. cover rest breakaway, low-speed sliding and nominal-speed sliding in both
   directions, at at least three loads;
3. recover applied resisting generalized torque from each engine's solver or a
   validated state-response calculation, not from the configured number;
4. verify all-zero produces zero friction contribution in both engines;
5. show that the legacy direct-number friction response changes with load in
   PhysX while MuJoCo keeps a fixed generalized-force bound. This test is
   expected to reject exact equivalence;
6. retain every row, environment version and source SHA. A summary without the
   raw probe table cannot advance the gate.

This stage requires no policy training and must not connect to the robot.

### P1: physical calibration evidence

The machine preregistration fixes the minimum evidence shape:

- all 31 actuated hinge joints, unless a joint grouping is first justified by
  residual equivalence;
- both directions; breakaway, low-speed and nominal-speed regimes; at least
  three distinct loads and five repeats per cell;
- joint state, acceleration, commanded/measured torque, estimated transmitted
  load and its uncertainty, temperature, controller mode, firmware, payload/
  pose and source session on every sample;
- complete-session train/holdout splitting. Adjacent rows from one sweep may
  not cross the split;
- numeric repeatability, fit and cross-engine tolerances frozen after a
  repeatability report but before adapter fitting.

This document authorizes no real command. Data must come from an approved
vendor/bench artifact or a separately reviewed G07 procedure after dry-run,
joint order, command scaling, effort envelope, safe halt and operator approval.

### P2: one physical model, two adapters

Fit the simplest candidate that meets the frozen session-held-out tolerance:

1. constant breakaway + Coulomb + viscous;
2. load-affine breakaway + Coulomb + viscous;
3. monotone load-table breakaway + Coulomb + viscous.

Static breakaway and dynamic friction must be non-negative, dynamic friction
must not exceed breakaway on the supported grid, viscous terms must be
non-negative, and positive/negative directions remain explicit. Load, speed,
pose or temperature outside the fitted envelope is out-of-contract, not an
invitation to extrapolate silently.

Both adapters bind the same latent-model SHA and support envelope:

- PhysX may use its native transmitted-force coefficient only if it passes all
  held-out probe cells; otherwise it needs a versioned explicit generalized-
  friction adapter;
- MuJoCo may use native `frictionloss` plus damping only if that passes the same
  cells; otherwise it also needs a versioned explicit adapter;
- engine-specific parameter bytes are expected. Acceptance is physical-model
  residual plus pairwise cross-engine residual, not parameter equality.

### P3: version the execution contract

Current schema-3 and BankExam code deliberately accepts non-zero PhysX friction
only through an inexact direct-number proxy. Do not weaken that guard for `SC`.
Before `SC` launch, choose and test one explicit versioning path:

- bump the training execution schema (for example, schema 4) to bind the latent
  model, support envelope, adapter kind/source/parameters and probe report; or
- bind a separate `plant_contract_schema_version=1` SHA from the training hard
  contract and ONNX, while old schema-3 judges reject non-zero exact claims.

Either path must make old exporters/evaluators fail closed, instantiate the
adapter before environment construction, record the realized 31-joint plant,
and require a new evaluator version. Merely adding optional JSON keys to an old
contract is not sufficient.

### P4: minimal high-information training axis

Freeze shared face, one schema-2 motion family, one schema-3 train/exam family,
reward/timing recipe, code and engine. Vary only training plant:

- `Z`: zero friction;
- `C`: calibrated plant.

Use two paired from-scratch seed blocks: `2 plants x 2 seeds = 4` training arms.
An old `SZ` checkpoint is not a matched control if adding `SC` changes code,
asset, solver or any non-plant hard-contract field; launch a contemporaneous
fresh `Z` arm instead.

At every accepted milestone, evaluate every policy under both `Z/C` plants in
both engines: `4 policies x 2 eval plants x 2 engines = 16` cells, on the same
immutable question order. `SP/LP` are excluded. The primary paired contrasts
are:

1. `C-train/C-eval - Z-train/C-eval`: does calibrated training fix deployment-
   plant quality and falls?
2. `C-train/Z-eval - C-train/C-eval`: how plant-specific is the learned policy?
3. `Z-train/C-eval - Z-train/Z-eval`: reproduce or resolve the 2026-07-07
   transfer failure under the repaired plant;
4. Isaac minus MuJoCo for the identical policy, eval plant, seed, checkpoint and
   schedule: does engine ranking agree?

Checkpoint policy stays unchanged: q10 is screen-only and cannot stop/promote;
q50 (100 questions, 50 per side) is the decision paper. Keep the best finite
checkpoint and do not wait for terminal if an earlier q50 peak wins.

## Fail-closed rules

Do not launch or promote `SC` if any of these is true:

- any required dataset, split, tolerance, latent-model, adapter, engine, asset,
  joint-order or probe SHA is absent;
- `SP/LP` is renamed or reused as calibrated `C`;
- a PhysX coefficient is copied into MuJoCo `frictionloss`;
- the current zero-only judge is made to label non-zero friction exact;
- adapter tuning sees held-out sessions or BankExam outcomes;
- any cell has missing raw rows, censoring, NaN/Inf, mixed papers or contract
  mismatch;
- a result exists in only one engine;
- a probe or evaluation leaves the calibrated support envelope;
- fresh `SC` lineage is claimed from a legacy/resumed checkpoint;
- q10 is used to choose the plant.

Even after all plant gates pass, G07 command ACK/timeout, mount calibration,
effort envelope and operator safe-halt remain independent deployment blockers.

## Reproducible local validation

```bash
python3 scripts/validate_phase1_plant_semantics_prereg.py \
  configs/phase1_plant_semantics_repair_prereg_20260711.json \
  --verify-repository-baseline

python3 -m pytest -q \
  tests/test_validate_phase1_plant_semantics_prereg.py
```

Expected current result: `PLANT_SEMANTICS_PREREG_OK`, status
`blocked_on_calibration_evidence`, four minimum fresh training arms, sixteen
paired evaluations per milestone, and `hardware_commands_authorized=false`.
