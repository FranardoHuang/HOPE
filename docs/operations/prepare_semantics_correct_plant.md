# Prepare a Semantics-Correct Plant Contract

Status: **offline contract tooling is ready; calibration evidence and simulator integration are
blocked** (2026-07-12).

This operation prepares future `SC` plant metadata. It does not touch the current Phase-1 training
checkout, start Isaac/MuJoCo, signal a Pod process or authorize a real-robot command.

Read first:

- [`../interfaces/plant_semantics_contract.md`](../interfaces/plant_semantics_contract.md)
- [`../research/phase1_plant_semantics_repair_2026-07-11.md`](../research/phase1_plant_semantics_repair_2026-07-11.md)
- `configs/phase1_plant_semantics_repair_prereg_20260711.json`

## Preconditions

Do not create a runtime-ready draft until all of these artifacts exist and have lower-case SHA-256
bindings:

1. calibration dataset manifest and complete-session train/holdout split;
2. repeatability report and numeric threshold contract frozen before adapter fitting;
3. selected latent model and support envelope for load, speed, temperature and pose;
4. independently fitted PhysX and MuJoCo adapter reports;
5. exact engine versions, assets, solver/integrator and rate contracts; the final MuJoCo asset is
   the Agibot vendor `a3_pingpong.xml`, not a generic reconstruction;
6. one immutable generalized-resistance probe schedule, two passed runtime probe reports and one
   passed cross-engine equivalence report;
7. the Gate3/Gate3B runtime-source SHA and a raw report proving the adapter was instantiated on all
   31 vendor-runtime joints;
8. canonical 31-joint A3 order.

The 2026-07-07 zero/non-zero policy probe has no raw-artifact SHA and is not calibration input.
The current `SP/LP` numbers are not adapter parameters. Do not copy them into a draft.

The v1 preregistration is also source-stale by design on current main. Its eight
source hashes are reproducible together at `d4ca566`, but the later strict
face179 change modified `training_contract.py`. Therefore
`validate_phase1_plant_semantics_prereg.py ... --verify-repository-baseline`
must currently return exit 2. Do not edit hashes merely to regain green: create
a new reviewed preregistration with the complete current closure before any
semantics-correct `SC` launch.

## Draft rules

Create a JSON object following plant-contract schema v1. Units are exact strings:

- PhysX coefficient: `dimensionless`;
- MuJoCo friction loss: `N*m`;
- MuJoCo damping: `N*m*s/rad`;
- load/speed/temperature support: `N*m`, `rad/s`, `degC`.

Both adapters must declare `source_parameter_origin` as
`engine_specific_fit_to_shared_latent_model`, bind the same latent-model/threshold/probe SHAs, and
use distinct engine-specific fit reports. The MuJoCo adapter must also declare
`runtime_target=agibot_vendor_mujoco_gate3_gate3b`; its `asset_sha256`,
`runtime_source_sha256` and `runtime_instantiation_report_sha256` bind the vendor MJCF and Gate3
runtime. Set:

```json
{
  "status": "ready_for_semantics_correct_runtime",
  "lineage_role": "fresh_semantics_correct_calibrated_plant",
  "hardware_commands_authorized": false,
  "legacy_direct_number_proxy": false
}
```

The compiler rejects unknown fields rather than ignoring them. The interface document lists the
complete schema; there is intentionally no checked-in fake ready contract with guessed A3
parameters.

## Bind and verify

From the repository root:

```bash
python3 scripts/compile_semantics_correct_plant_contract.py \
  bind /path/to/reviewed_draft.json /path/to/bound_plant_contract.json

python3 scripts/compile_semantics_correct_plant_contract.py \
  verify /path/to/bound_plant_contract.json
```

`bind` canonicalizes only for hashing; it does not repair values. It inserts the payload SHA and
then runs the same full validator. Expected success tokens are `PLANT_CONTRACT_BIND_OK` and
`PLANT_CONTRACT_VERIFY_OK`, each followed by `hardware_commands_authorized=false`.

Any edit after binding invalidates `contract_sha256`. Missing evidence, wrong units, NaN/Inf,
negative native parameters, a non-passed probe/equivalence report, a mismatched joint order, a
non-independent fit report or unsupported backend fails closed.

## Prepare one engine adapter

The requested-support JSON has the same shape as the contract's support block:

```json
{
  "load_abs_Nm": [0.0, 100.0],
  "speed_abs_rad_s": [0.0, 12.0],
  "temperature_C": [20.0, 35.0],
  "pose_ids": ["ready_stand", "forehand_contact", "backhand_contact"],
  "pose_ids_sha256": "<canonical SHA of the ordered pose_ids array>"
}
```

Prepare PhysX and the Agibot vendor Gate3/Gate3B MuJoCo adapter separately:

```bash
python3 scripts/compile_semantics_correct_plant_contract.py prepare \
  /path/to/bound_plant_contract.json \
  --engine physx \
  --requested-support /path/to/requested_support.json \
  --output /path/to/physx_runtime_adapter.json

python3 scripts/compile_semantics_correct_plant_contract.py prepare \
  /path/to/bound_plant_contract.json \
  --engine mujoco \
  --requested-support /path/to/requested_support.json \
  --output /path/to/mujoco_runtime_adapter.json
```

An envelope extending beyond calibration fails. These commands only compile metadata; they do not
load the values into either engine.

## Local verification

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_plant_contract_v1.py \
  tests/test_validate_phase1_plant_semantics_prereg.py
```

The current expected state is: compiler and historical-snapshot tests pass;
current-checkout baseline verification fails closed on `training_contract.py`;
the preregistration remains `blocked_on_calibration_evidence`; no `SC` arm or
non-zero exact evaluator exists.

## Runtime integration still required

Before any `SC` launch, a separate reviewed change must:

1. bind `plant_contract_schema_version=1`, plant SHA and prepared adapter SHA into the training hard
   contract, checkpoint and ONNX;
2. instantiate the adapter before environment construction and compare all 31 realized values to
   the prepared artifact;
3. add a new evaluator profile that consumes the MuJoCo adapter rather than
   `--allow-inexact-contract` or `contract-proxy`;
4. instantiate that adapter in the vendor Gate3/Gate3B runtime, verify all 31 realized joint values,
   run the same physical probe schedule in both engines and retain raw rows;
5. preregister the fresh `Z/C x two paired seeds` launch and its 16 same-paper evaluations per
   milestone.

Until those steps pass, leave current `SZ/SP/LZ/LP`, trainers and workers untouched.
