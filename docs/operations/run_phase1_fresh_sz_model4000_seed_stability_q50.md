# Fresh SZ model_4000 four-seed matched q50 queue

Status: preregistered and activation-ready. Both Pod audits, the all-four activation and both
read-only `contract-check` calls completed on 2026-07-13 local time; no q50 `prepare`, judge,
trainer signal, or robot command has run. This queue exists to
separate delayed seed4 learning from persistent seed4 weakness at the next matched
checkpoint. It does not authorize a training change, checkpoint promotion, deployment, or
hardware.

The final behavioral arbiter remains the Agibot vendor MuJoCo Gate3/Gate3B runtime. This
clean K100 MuJoCo paper is an earlier checkpoint/seed instrument, not a substitute for that
gate.

## Frozen question and known limitation

The four fresh formal-target `SZ` seeds use `model_4000.pt` and the **byte-identical** K100
paper used for the model-2000 stability result:

- schedule file SHA-256 `66e89986a2b726d529179fcb4c745625ebed0380d59664caceefc55e86071cb3`;
- schedule semantic SHA-256 `7dc6af822fb4130b8c324843f179d77f882d1326306bb19802b00f94447dff3e`;
- question-order SHA-256 `b87e81a34ff2d31766e17345f0a8c9d77665b78874093e26bdae257e8ed21f91`;
- exact-family exam-bank SHA-256 `d7db2568beee990ef1d64b2dce9f0ab56ca76377f8993d820b6388292d0f5096`;
- 100 uncensored attempts, 50 per side, seed 0, noise 0, hold `[0,100]`, one-question
  reset, no wrap, and no inexact escape.

No new schedule may be materialized. Family, training recipe, hard contract, four seeds and
the model-2000 stability thresholds are unchanged:

| check | unchanged threshold |
| --- | ---: |
| four-seed median aggregate | `>= 0.75` |
| worst-seed aggregate | `>= 0.65` |
| best-minus-worst aggregate | `<= 0.20` |
| every seed on every side | `>= 0.50` |

Two facts were known before this preregistration. At model 2000, seeds 1--4 scored
`.83/1.00/1.00/.20`, so the stability gate failed. Seed1 model 4000 had already scored
`.50` aggregate (`FH=.00`, `BH=1.00`) on this exact paper. Therefore the model-4000
four-seed **family stability gate is already mathematically unable to pass** the unchanged
worst-seed threshold. This paper still answers the narrower, preregistered question:

- seed4 supports delayed learning only if its 4k aggregate is at least `.65` and both sides
  are at least `.50`—the same old thresholds, not hindsight-tuned ones;
- otherwise seed4 weakness remains persistent through 4k;
- even if seed4 recovers, do not claim the family is stable because known seed1 4k is `.50`.

The preregistration allowed seed1 reuse only after full raw-chain revalidation. The reviewed
execution contract chooses the more conservative route: seed1 is rerun on the identical K100
bytes and prior score reuse is rejected. Its known checkpoint SHA is `1a8fcf3d...e9071`; a
different discovered SHA still fails the readiness audit.

## Content-bound source

- preregistration:
  `configs/phase1_fresh_SZ_model4000_seed_stability_q50_prereg_20260712.json`, SHA-256
  `ca5ea90f8420ef4c96ee05881b25d062cc437faa97510babca45299afcabbff0`;
- offline queue:
  `configs/phase1_fresh_SZ_model4000_seed_stability_q50_queue_20260712.json`, SHA-256
  `d4e69d91adfe7a42aee897c11b1b6d6bf7e5eaa7fb81d856b66cab7b3f7d3909`;
- readiness validator:
  `scripts/validate_phase1_fresh_sz_model4000_q50_queue.py`, SHA-256
  `e763ecb9a822f7e1c2e9338749701fcd4bfea9f26f9b6fe5b4b189f8ca5a6cd3`;
- bound fresh exact validator SHA-256
  `3528250777a170791f39d8dd17716c2a7f8ca91416a3ffa8433ec5eb691ed9e0`;
- activation-consuming runner:
  `scripts/run_phase1_fresh_sz_model4000_q50.py`, SHA-256
  `de0abff6096efdea8ce78dbac6f3115d09e70be8ca0fc841a36be2cdbfbf6b85`;
- independent execution contract:
  `configs/phase1_fresh_SZ_model4000_seed_stability_q50_execution_20260712.json`, SHA-256
  `3109acd41726ef1a3063637e2a565cb2f4abe8992bb96473940700981e7c4385`.

The readiness validator deliberately exposes only `validate-config`, `audit-pod`, and
`activate`. It imports no process-control module and has no SSH, judge launch, kill, or
signal path. The queue has `runtime_entrypoint=null`; an activation artifact does not start
a judge. The separate runner keeps that historical queue byte-identical and requires the exact
activation path plus caller-supplied file SHA on every `contract-check`, `prepare`, `run`, and
`aggregate` invocation. It revalidates both Pod audits and all four embedded finite/iteration/
contract/lineage audits before accepting the activation. On a Pod it additionally rehashes and
re-audits that Pod's two live checkpoint files and adjacent hard contracts.

## Mandatory all-four barrier

The queue remains runtime-ineligible until **both** Pod audits are content-bound and their
union covers seed1/2/3/4 exactly. Every arm must satisfy all of the following:

1. the path ends in `model_4000.pt` and the checkpoint embeds `iter=4000`;
2. every floating tensor is finite;
3. embedded schema version is 3 and `training_contract_lineage_exact=true`;
4. embedded contract SHA equals the adjacent contract SHA
   `3a3b3d95...b9972`;
5. the adjacent hard contract remains fresh `SZ`: `shared_plus_y`, motion exact, schema-3
   bank family exact, and 31/31 zero friction coefficients;
6. both clean checkouts stay at training `6d93bcb...` and eval `46a0ce2...`;
7. both Pod audits bind the same schedule file/semantic/question-order SHA.

One Pod audit is explicitly insufficient and says
`runtime_authorized_by_this_pod_audit=false`. Only `activate` can combine both audits; the
result still says `judges_started=0` and only permits a future runner to prepare against
those frozen checkpoint hashes. No trainer or checkpoint worker may receive a signal.

## Validation now

Run locally or in an isolated clean control copy; this performs no Pod access:

```bash
python3 scripts/validate_phase1_fresh_sz_model4000_q50_queue.py \
  --queue configs/phase1_fresh_SZ_model4000_seed_stability_q50_queue_20260712.json \
  --expected-queue-sha256 d4e69d91adfe7a42aee897c11b1b6d6bf7e5eaa7fb81d856b66cab7b3f7d3909 \
  validate-config

pytest -q tests/test_validate_phase1_fresh_sz_model4000_q50_queue.py
pytest -q tests/test_run_phase1_fresh_sz_model4000_q50.py
```

Accepted source verification on 2026-07-12: queue/barrier plus runner focused suite `40 passed`;
committed queue validation prints `PASS; no runtime`. This is source evidence only.

## Partial readiness execution (2026-07-12)

The first fail-closed runtime step is partially complete:

- both Pods have the exact seven-file source/config bundle at
  `/workspace/codexschema/phase1_fresh_20260711/control/SZ_model4000_seed_stability_q50_v1/source_d67310f`;
- both Pods have the same pre-existing K100 bytes at
  `/workspace/codexschema/phase1_fresh_20260711/control/SZ_model4000_seed_stability_q50_v1/shared_clean_k100.schedule.json`;
- all reviewed source/config hashes and schedule file/semantic/order hashes matched;
- Pod2 seed2/seed4 produced `pod2_ready_audit.json`, file SHA-256
  `4f25786b7524db848b9adebf5a8946bb8f82280ea8d1d5a1243ae85533f565f7` and embedded content
  SHA-256 `5df5f2995149c168a90bce3cf662b53322d9fbca9da4b724b814821f2c9bdb11`; the exact bytes are
  checked in as
  `configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod2_ready_audit_20260712.json` and the
  local relay remains at `/private/tmp/phase1_model4000_activation_relay/pod2_ready_audit.json`.

Pod1 accepted the source/K100 deployment, but each later `audit-pod` attempt stopped at SSH
handshake timeout before a remote command began. Treat Pod1 process/checkpoint state for those
attempts as unknown, not failed. Do not rerun Pod2, invent a second schedule path, activate from one
audit, prepare or launch judges. Resume only the no-clobber Pod1 audit after a low-frequency clean
connection, then combine both observed audit bytes exactly as described below.

Verify the preserved Pod2 evidence without accessing a Pod:

```bash
A=configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod2_ready_audit_20260712.json
sha256sum "$A"
jq -cS '.content' "$A" | tr -d '\n' | sha256sum
```

The two expected digests are respectively `4f25786b...565f7` and
`5df5f299...bdb11`. This still cannot authorize `activate` without the separately observed Pod1
artifact.

## Future readiness audit and activation

Do this only after the ordinary monitor reports every accepted model-4000 checkpoint present.
Preserve the repo-like `configs/` and `scripts/` layout in an external clean control copy;
do not alter either frozen training checkout.

On Pod1, audit seed1/seed3. On Pod2, audit seed2/seed4. `$SOURCE_SCHEDULE` must point to the
same pre-existing file bytes on both hosts, and `$QUEUE_SHA` is the queue SHA above:

```bash
python3 scripts/validate_phase1_fresh_sz_model4000_q50_queue.py \
  --queue configs/phase1_fresh_SZ_model4000_seed_stability_q50_queue_20260712.json \
  --expected-queue-sha256 "$QUEUE_SHA" \
  audit-pod \
  --pod pod1 \
  --schedule-source "$SOURCE_SCHEDULE" \
  --output /workspace/codexschema/phase1_fresh_20260711/q50/fresh_SZ_model4000_seed_stability_q50_barrier_v1/pod1_ready_audit.json
```

Use the identical command with `--pod pod2` and the preregistered Pod2 output. These commands
only hash and inspect checkpoints/contracts and write no-clobber readiness JSON; they start no
judge. If any arm is absent, non-finite, wrong-iteration, wrong-lineage or wrong-contract, keep
the failure output/log and wait—do not partially start the other seeds.

After copying both immutable audits to one control host, combine them only with their observed
file SHAs:

```bash
python3 scripts/validate_phase1_fresh_sz_model4000_q50_queue.py \
  --queue configs/phase1_fresh_SZ_model4000_seed_stability_q50_queue_20260712.json \
  --expected-queue-sha256 "$QUEUE_SHA" \
  activate \
  --pod1-audit "$POD1_AUDIT" \
  --pod1-audit-sha256 "$POD1_AUDIT_SHA" \
  --pod2-audit "$POD2_AUDIT" \
  --pod2-audit-sha256 "$POD2_AUDIT_SHA" \
  --output-dir /workspace/codexschema/phase1_fresh_20260711/control/SZ_model4000_seed_stability_q50_v1/activation
```

## Activation-consuming execution

Do not improvise a judge command after activation. Deploy the runner, execution config, queue,
preregistration, queue validator and pinned fresh helper together in the same repo-like external
control copy. Do not modify either frozen train/eval checkout. The activation and both Pod audit
files must remain at the absolute paths recorded inside the activation; copy the immutable bytes
to each Pod control namespace when necessary. `$SOURCE_SCHEDULE` must be the exact absolute path
recorded in `activation.content.shared_schedule.path`; a same-hash file at a different path is
rejected rather than silently substituted.

Set the content hashes from the reviewed source and the actual activation bytes:

```bash
CONFIG=configs/phase1_fresh_SZ_model4000_seed_stability_q50_execution_20260712.json
CONFIG_SHA=3109acd41726ef1a3063637e2a565cb2f4abe8992bb96473940700981e7c4385
RUNNER=scripts/run_phase1_fresh_sz_model4000_q50.py
ACTIVATION=/absolute/path/from/activate/activation_<content-sha>.json
ACTIVATION_SHA=$(sha256sum "$ACTIVATION" | awk '{print $1}')

python3 "$RUNNER" \
  --config "$CONFIG" --expected-config-sha256 "$CONFIG_SHA" \
  --activation "$ACTIVATION" --expected-activation-sha256 "$ACTIVATION_SHA" \
  contract-check --pod pod1 --schedule-source "$SOURCE_SCHEDULE"
```

Run the same read-only check for Pod2. It writes nothing and launches nothing. Only after it
passes, prepare each Pod's fixed no-clobber state directory:

```bash
python3 "$RUNNER" \
  --config "$CONFIG" --expected-config-sha256 "$CONFIG_SHA" \
  --activation "$ACTIVATION" --expected-activation-sha256 "$ACTIVATION_SHA" \
  prepare --pod pod1 --schedule-source "$SOURCE_SCHEDULE"
```

`prepare` copies—not materializes—the exact schedule bytes and writes
`runtime_contract.activation_bound.prepared.json`. Record its observed SHA. `run` accepts only
that exact runtime contract:

```bash
RUNTIME=/workspace/codexschema/phase1_fresh_20260711/q50/fresh_SZ_model4000_seed_stability_q50_pod1_v1/runtime_contract.activation_bound.prepared.json
RUNTIME_SHA=$(sha256sum "$RUNTIME" | awk '{print $1}')

python3 "$RUNNER" \
  --config "$CONFIG" --expected-config-sha256 "$CONFIG_SHA" \
  --activation "$ACTIVATION" --expected-activation-sha256 "$ACTIVATION_SHA" \
  run --pod pod1 --runtime-contract "$RUNTIME" \
  --expected-runtime-contract-sha256 "$RUNTIME_SHA"
```

Repeat for Pod2 with its configured runtime path. Each Pod runs exactly two arms serially
(Pod1 seed1 then seed3; Pod2 seed2 then seed4). Seed1 is a fresh identical-paper judge, never a
reused result. Every `judge.sh` child starts a new session, records exact PID=PGID, sets
`JUDGE_KIT_BOOT_LOCK=/workspace/.kit_boot.lock`, and waits to completion before the next seed.
The runner has no SSH or signal API. Any nonzero exit, missing report, or raw-chain validation
failure leaves the per-seed state and log in place and prevents a Pod result.

After copying the two immutable Pod-result JSON files to the configured control host, aggregate
only with their observed file SHAs:

```bash
python3 "$RUNNER" \
  --config "$CONFIG" --expected-config-sha256 "$CONFIG_SHA" \
  --activation "$ACTIVATION" --expected-activation-sha256 "$ACTIVATION_SHA" \
  aggregate \
  --pod1-result "$POD1_RESULT" --pod1-result-sha256 "$POD1_RESULT_SHA" \
  --pod2-result "$POD2_RESULT" --pod2-result-sha256 "$POD2_RESULT_SHA"
```

The Pod run revalidates `evaluation_contract_exact=true`, 100 uncensored rows, 50 per side,
schedule/order, MJCF, execution/ready-state SHA, checkpoint/hard-contract SHA, report, summary
and attempt-ledger bytes before writing its result. Aggregation rechecks the two result content
hashes and all four bindings. It always reports `family_stable_claim_allowed=false` because the
known pre-registered seed1 4k score is `.50`; it classifies seed4 as delayed learning only at
aggregate `>=.65` and both sides `>=.50`, otherwise persistent weakness through 4k. Neither
classification stops training, promotes a checkpoint, deploys, or authorizes hardware.

## Materialized barrier evidence (2026-07-13 local / 2026-07-12 UTC)

The exact committed bundle is present on both Pods at:

`/workspace/codexschema/phase1_fresh_20260711/control/SZ_model4000_seed_stability_q50_v1/source_d67310f`

The common immutable schedule is present on both Pods at:

`/workspace/codexschema/phase1_fresh_20260711/control/SZ_model4000_seed_stability_q50_v1/shared_clean_k100.schedule.json`

Its SHA remains `66e89986...71cb3`; the bundle's queue/prereg/validator/helper/runner/config SHAs
match the values above. Readiness artifacts are byte-identical at their recorded absolute paths
on both Pods:

- Pod1 seed1/seed3 audit:
  `/workspace/codexschema/phase1_fresh_20260711/q50/fresh_SZ_model4000_seed_stability_q50_barrier_v1/pod1_ready_audit.json`,
  file SHA `3fc325e1edce6b8e6570cfcbbd4308b168d4a646c2b098061cf4b155fcd247b8`,
  content SHA `5f378181147fbd4780974ce9155d1561fd9c59da8aaed858b6b3c8daa2aaa1dd`;
- Pod2 seed2/seed4 audit:
  `/workspace/codexschema/phase1_fresh_20260711/q50/fresh_SZ_model4000_seed_stability_q50_barrier_v1/pod2_ready_audit.json`,
  file SHA `4f25786b7524db848b9adebf5a8946bb8f82280ea8d1d5a1243ae85533f565f7`,
  content SHA `5df5f2995149c168a90bce3cf662b53322d9fbca9da4b724b814821f2c9bdb11`.

The all-four activation is:

`/workspace/codexschema/phase1_fresh_20260711/control/SZ_model4000_seed_stability_q50_v1/activation/activation_eaa92ca201c4cd85c81b190fc2aee3b01ec4f6a2e70383eca6637373f87aa4fb.json`

Its file SHA is `9dea76c2a9039dc35f8f996fa112e0e28ee320cb9b7c7ec877be942e021ce704`
and content SHA is `eaa92ca201c4cd85c81b190fc2aee3b01ec4f6a2e70383eca6637373f87aa4fb`.
It covers seed1--4 exactly and records `judges_started=0`.

The byte-identical Pod1 audit and activation are checked in at
`configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod1_ready_audit_20260713.json` and
`configs/phase1_fresh_SZ_model4000_seed_stability_q50_activation_20260713.json`; the already
preserved Pod2 audit remains
`configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod2_ready_audit_20260712.json`.

Verify each preserved file and its embedded canonical content independently:

```bash
for A in \
  configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod1_ready_audit_20260713.json \
  configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod2_ready_audit_20260712.json \
  configs/phase1_fresh_SZ_model4000_seed_stability_q50_activation_20260713.json; do
  sha256sum "$A"
  jq -cS '.content' "$A" | tr -d '\n' | sha256sum
done
```

Both Pods passed the activation-consuming runner's `contract-check`. Immediately afterward each
reported no child judge, MuJoCo evaluator, play/Kit process or holder of
`/workspace/.kit_boot.lock`. This authorizes the next explicit `prepare` step under the frozen
contract; it is not itself preparation or permission to bypass a fresh pre-run conflict check.
No training/eval checkout was written and no signal was sent.
