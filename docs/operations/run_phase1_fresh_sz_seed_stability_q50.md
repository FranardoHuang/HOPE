# Fresh SZ model_2000 four-seed q50 stability paper

Status: completed and archived on 2026-07-11; the preregistration preceded every new judge.
This paper is checkpoint/seed-stability evidence only. It does not stop or promote any
training arm, authorize deployment, close the
calibrated-plant or recovery/continuous-rally gates, or authorize a real-robot command.
The final behavior gate remains the AgiBot vendor MuJoCo Gate3/Gate3B path; Isaac is an
accompanying diagnostic, not the primary acceptance instrument.

## Frozen question

Evaluate fresh formal-target cell `SZ` (`v4rg_runtime_order_v3`, `shared_plus_y`, 31/31
zero joint-friction coefficients) at `model_2000.pt` for training seeds 1--4. Every seed gets
the same already materialized schema-3 K100 paper:

- schedule file SHA-256: `66e89986a2b726d529179fcb4c745625ebed0380d59664caceefc55e86071cb3`
- schedule semantic SHA-256: `7dc6af822fb4130b8c324843f179d77f882d1326306bb19802b00f94447dff3e`
- question-order SHA-256: `b87e81a34ff2d31766e17345f0a8c9d77665b78874093e26bdae257e8ed21f91`
- exam-bank SHA-256: `d7db2568beee990ef1d64b2dce9f0ab56ca76377f8993d820b6388292d0f5096`
- 50 questions per side, schedule seed 0, noise 0, hold range `[0,100]`, one-question
  reset, no wrap, and no inexact escape

The source paper was materialized before any q50 outcome and selected without the seed2/3/4
policy or outcomes. New materialization is forbidden; identical source bytes are copied to
both Pod state roots. The four q10 values (`.90/1.00/1.00/.25`) are only the trigger for this
paper and remain screen-only.

The preregistered stability checks are:

- median aggregate return rate at least `0.75`
- worst-seed aggregate return rate at least `0.65`
- best-minus-worst aggregate return rate at most `0.20`
- every seed on every side at least `0.50`

These thresholds ask whether the checkpoint evidence is both useful and repeatable. They are
not a deploy threshold. Pass closes only this model_2000 seed-stability question; failure keeps
it open. Either outcome leaves all arms training unchanged.

## Recorded result

All four results reproduced the frozen exact paper with 100 uncensored attempts, 50 per side,
matching schedule/question order, matching MuJoCo execution and ready-state hashes, finite
checkpoint audits, and zero recorded physical falls:

| seed | forehand | backhand | aggregate |
| --- | ---: | ---: | ---: |
| 1 | 33/50 | 50/50 | 83/100 |
| 2 | 50/50 | 50/50 | 100/100 |
| 3 | 50/50 | 50/50 | 100/100 |
| 4 | 0/50 | 20/50 | 20/100 |

The median aggregate rate was `0.915` and passed its `0.75` check. The worst seed was `0.20`,
the best-minus-worst spread was `0.80`, and the worst seed/side rate was `0.00`; those three
checks failed their preregistered `0.65`, `0.20`, and `0.50` limits. The formal outcome is
`fail_seed_stability_checkpoint_evidence`. In particular, the three strong seeds may not be
used to hide seed4. This is consistent with seed4's earlier direction-only q10 (`0.25`) and
turns that warning into a full-paper stability failure.

Archived evidence:

- Pod1 result file SHA-256: `0e651edae4e0e237e51ed2445f36e8dcc6903dea6444ad532f13c2819128eedc`
- Pod2 result file SHA-256: `ad1187a9a24707fe184a4751b73be05e43d46ac5596590ae3e598355f3c39bd4`
- aggregate content SHA-256: `a756bf1d0e76d1016992ae241b935cf92b3c84ffd55fe503e7c199626d9c8ffd`
- aggregate file SHA-256: `d856468fb93461be52498a24655b25993ce28f530f5989e61110e33421736e43`

The recorded action is therefore: keep this seed-stability question open, continue every
training arm unmodified, send no trainer/worker signal, and keep all downstream behavior,
plant, recovery, deployment, and real-robot gates open.

The next matched milestone is separately preregistered at model 4000 in
`docs/operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md`. It reuses these exact
K100 bytes and thresholds. Because seed1 model 4000 was already known to score `0.50`, that
paper can distinguish delayed versus persistent seed4 weakness but cannot retroactively pass
four-seed family stability. Its all-four finite/iteration/contract/lineage activation barrier
must pass before any new q50 runtime.

## Content-addressed chain

The immutable inputs are:

- preregistration: `configs/phase1_fresh_SZ_model2000_seed_stability_q50_prereg_20260711.json`
- execution contract: `configs/phase1_fresh_SZ_model2000_seed_stability_q50_execution_20260711.json`
- runner: `scripts/run_phase1_fresh_sz_seed_stability_q50.py`
- runner SHA-256: `f023c0debb013685af03922cb848438e8edf78081419440cdab0b8a6cccfa3ad`
- preregistration SHA-256: `cf3ce857b1b1a688e808e4a85ea662b252e84d2ca32240c29ee1b08009aaeabd`
- execution-contract SHA-256: `85a0101704b96eaa044c607ee62db9a40b3cb59186a8ab626e389887008991ff`

The runner imports the previously audited fresh exact q50 validator only when its SHA is
`3528250777a170791f39d8dd17716c2a7f8ca91416a3ffa8433ec5eb691ed9e0`.
It has no SSH or process-signal surface. Each local judge starts in its own recorded process
group and runs through `judge.sh` plus the shared Kit lock. Never use a broad kill command.

Seed1 may reuse its earlier K100 result only after the runner revalidates the old runtime
contract, identical schedule bytes, checkpoint and adjacent hard-contract bindings, raw report,
summary, 100-row uncensored attempt ledger, all denominators, exactness, and checked-in result
binding. `reuse-check` is read-only. If it fails, preserve the evidence and rerun seed1 on the
identical paper in a fresh no-clobber state directory.

## Safe execution order

Copy the committed runner, execution contract, preregistration, the bound fresh validator, and
the checked seed1 result into the external control directory. Do not copy them into either
training checkout. Keep both training checkouts clean at
`6d93bcb16c422a2f42748c2dc99432559653480b` and both independent evaluation checkouts clean at
`46a0ce24524fdb843e55fe82ba4c045f2adc090f`.

Use the copied absolute paths below as `$RUNNER`, `$CONFIG`, and `$PREREG`, and bind the
execution config SHA on every invocation:

```bash
python3 "$RUNNER" \
  --config "$CONFIG" \
  --expected-config-sha256 85a0101704b96eaa044c607ee62db9a40b3cb59186a8ab626e389887008991ff \
  --preregistration "$PREREG" \
  contract-check --pod pod1 --schedule-source "$SOURCE_SCHEDULE"

python3 "$RUNNER" \
  --config "$CONFIG" \
  --expected-config-sha256 85a0101704b96eaa044c607ee62db9a40b3cb59186a8ab626e389887008991ff \
  --preregistration "$PREREG" \
  prepare --pod pod1 --schedule-source "$SOURCE_SCHEDULE"
```

Run the same two phases for Pod2 using the byte-identical schedule. Record the printed runtime
contract SHA. Before Pod1 starts a judge, validate seed1 reuse:

```bash
python3 "$RUNNER" \
  --config "$CONFIG" \
  --expected-config-sha256 85a0101704b96eaa044c607ee62db9a40b3cb59186a8ab626e389887008991ff \
  --preregistration "$PREREG" \
  reuse-check \
  --runtime-contract "$POD1_RUNTIME" \
  --expected-runtime-contract-sha256 "$POD1_RUNTIME_SHA"
```

Then run Pod1 (`seed1`, `seed3`) and Pod2 (`seed2`, `seed4`). The runner serializes the two
local arms; the two Pods may progress independently. `--rerun-seed1` is permitted only after a
failed reuse audit and only in a fresh no-clobber Pod1 state directory. Monitor exact recorded
judge PID/PGID, the Kit lock, GPU/RAM, logs, and result hashes. Do not signal trainers or
checkpoint workers.

After both content-addressed Pod results are copied to one control host, run `aggregate` with
both observed file SHAs. Preserve the aggregate file named by its content SHA. Regardless of
the recorded pass/fail, continue all training arms and keep vendor MuJoCo Gate3/Gate3B,
calibrated-plant, cross-instrument, recovery/continuous-rally, deployment, and real-robot gates
open.
