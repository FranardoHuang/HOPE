# Plant Semantics Contract

Status: **contract/compiler implemented; no calibrated A3 contract exists and no runtime is
wired** (2026-07-12).

This interface prevents a non-zero MuJoCo joint `frictionloss` value from being treated as an
Isaac/PhysX joint-friction coefficient. It is the future contract for the `SC` cell (shared face +
semantics-correct calibrated plant), not a modification of the current Phase-1 arms.

Implementation:

- `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/plant_contract.py`
- `scripts/compile_semantics_correct_plant_contract.py`
- `hope_training/whole_body_tracking/tests/test_plant_contract_v1.py`

## Quantities and units

| Quantity | Canonical unit | Runtime meaning |
| --- | --- | --- |
| PhysX native joint friction | `dimensionless` | load-dependent coefficient applied to transmitted spatial force |
| MuJoCo `frictionloss` | `N*m` | load-independent generalized dry-friction bound for an A3 hinge |
| MuJoCo damping | `N*m*s/rad` | velocity-linear passive generalized torque |
| latent-model load | `N*m` | absolute transmitted generalized load used by the calibrated model |
| joint speed | `rad/s` | absolute hinge speed |
| temperature | `degC` | calibration/runtime support variable |

There is no non-zero numeric conversion between `dimensionless` and `N*m`. The API
`zero_only_unit_conversion` accepts an identical-unit vector or an exact all-zero vector across
different units; every non-zero cross-unit request raises. Zero is the current common semantic
special case, but it is not evidence that a zero-friction plant matches A3.

## Ready contract v1

A ready contract is strict JSON with no unknown keys. Its canonical payload SHA excludes only the
self-referential `contract_sha256` field. It binds:

- exactly 31 unique A3 joint names and the canonical joint-order SHA;
- one preregistered physical latent-model family, explicit unit table, source-dataset manifest,
  complete-session split, repeatability report, frozen thresholds, model-selection report and
  calibrated support envelope;
- a common probe schedule and a passed cross-engine equivalence report;
- one independently fitted PhysX adapter and one independently fitted MuJoCo adapter, each with an
  engine version, asset, solver, integrator, physics/policy rate, adapter source, fit report and
  passed runtime-probe report;
- for the MuJoCo leg specifically, `runtime_target=agibot_vendor_mujoco_gate3_gate3b`, the vendor
  `a3_pingpong.xml` asset SHA, Gate3/Gate3B runtime-source SHA and a raw 31-joint runtime
  instantiation report SHA;
- `hardware_commands_authorized=false` and `legacy_direct_number_proxy=false`.

The currently implemented backends are deliberately narrow:

- PhysX `native_transmitted_force_coefficient`, with 31 explicitly dimensionless coefficients;
- Agibot vendor Gate3/Gate3B MuJoCo `native_frictionloss_plus_damping`, with separate 31-element
  `N*m` and `N*m*s/rad` vectors. A generic standalone MuJoCo wrapper cannot fill this final-runtime
  role.

The preregistration permits future explicit generalized-friction adapters, but contract v1 rejects
those backends until their parameter law and runtime integration have a reviewed schema. An
unrecognized backend cannot silently fall back to native fields.

## Runtime preparation

`prepare_runtime_adapter` revalidates the complete contract and requires a requested load, speed,
temperature and pose envelope. Every requested interval must be a subset of the calibrated support
and every pose ID must be listed in it. A prepared adapter carries the parent plant SHA, latent
model, joint order, engine/solver/asset/probe evidence, runtime source and 31-joint instantiation
report, explicit parameter units and its own `runtime_adapter_sha256`.

A prepared JSON is replay metadata only. It does not instantiate Isaac or MuJoCo, launch training,
declare BankExam exactness, promote a checkpoint, or authorize a robot command. Those integrations
require a new training/evaluator binding; current schema-3 continues to reject non-zero formal
MuJoCo parity.

## Claim boundary

- Current `SZ`: execution-protocol exact for the implemented all-zero cross-engine plant only;
  `deployment_plant_calibrated=false`.
- Current `SP/LP`: historical direct-number proxy; never acceptable as a contract-v1 calibrated
  adapter.
- Future `SC`: may be named only after a real contract-v1 artifact passes validation and the fresh
  paired training/evaluation preregistration is rebound.

Passing contract-v1 validation proves content-addressed adapter replay inside a calibrated support
envelope. It does not prove policy quality, continuous-rally behavior, hardware safety or
deployment readiness.

## Evaluator self-contact honesty

Isaac training disables robot self-collision, while the vendor MuJoCo model can resolve contacts
between robot collision geoms. A formal MuJoCo BankExam therefore cannot remain exact after any
such solver contact. The evaluator defines robot geoms as the explicit `pelvis_link` articulation
subtree, excluding world/table/net, dynamic balls, mocap helpers and unrelated free bodies, and
samples that set after **every physics substep**, not only at the policy/control rate. The first
formal robot-pair contact fails closed immediately. Diagnostic lanes may continue, but must retain
the number of physics substeps sampled, substeps with contact, total classified pairs, maximum
penetration and worst pair; a transient contact cannot be erased by a clean final substep.

This is an evaluator truthfulness rule, not a calibrated collision model and not evidence that the
Isaac and vendor contact solvers are behaviorally equivalent.
