# Fresh SZ model-4000 four-seed matched q50

Status: Preregistered; runtime prepared; execution blocked on supervisor merge/review and Linux
fake-runner smoke

Human owner: Franco

Executor: Codex

Branch: `Franco_codex/q50-persistent-supervisor`

## Question and decision scope

On the unchanged exact [q50/K100](../DEFINITIONS.md#q50-and-k100) paper, does seed4's model-4000
checkpoint support delayed learning, or is its weakness persistent through 4k? Known seed1 4k
performance already forbids a family-stable PASS, so this run may only classify seed4; it cannot
adopt a baseline, change training, deploy, or authorize hardware.

## Inputs and immutable bindings

The complete paper/checkpoint/threshold/source binding and both prepared-runtime evidence records
live in the
[model-4000 q50 operation](../operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md).
The startup wrapper adds no evaluation variable. Its exact new bindings are:

- supervisor source SHA `fd565ca453d95712fa7045fec77e8fce29e93f0a13590d72311fcb7719ab800e`;
- supervisor config SHA `d1a76a5756d32642136884b4414e9fad2b5a80a257bb9c6fdd9e61a1108f6050`;
- all-four activation file SHA `9dea76c2a9039dc35f8f996fa112e0e28ee320cb9b7c7ec877be942e021ce704`;
- Pod1 prepared runtime SHA `2b76a5a917c0a5d88ab5eec6b984b3d4ed2faa07484804bb42551f310378201e`;
- Pod2 prepared runtime SHA `dbecc102cdb388873c9369f60e3820a0f4c6949cc925cd5f3123731eec8d1c9b`;
- both Pods' Python realpath `/usr/bin/python3.10` and binary SHA
  `06630724486efc9d97db03c62949511584b896c110097153ef970f9294fd3ba0`.

## Design and controls

Pod1 remains seed1 then seed3; Pod2 remains seed2 then seed4. Each Pod's two judges are serial and
use the existing shared Kit lock. The supervisor only provides a two-phase no-clobber startup and
read-only identity/result inspection; it does not alter the old runner, environment paper,
checkpoints, schedule, denominators, reset/censoring policy or thresholds.

## Acceptance and failure rules

The unchanged rule classifies seed4 as delayed learning only at aggregate `>=.65` and both sides
`>=.50`; otherwise weakness is persistent through 4k. The family-stable claim is always false due
to known seed1 4k aggregate `.50`. Any binding mismatch, pre-existing result, incomplete startup
handshake, reused process identity, non-exact result or missing full bound-runner validation fails
closed and preserves evidence. First possible visibility of the parent commit token's final link is
irreversible, even if the following directory fsync reports an error. Lack of immediate
acknowledgment becomes `token_published_pending_ack`; a valid acknowledgment without immediate exact
exec becomes `committed_pending_exec`. Both return zero, preserve the fixed state directory and
never create retry authority, even after the old tokenless-startup deadline.

## Reproduction

Source tests and the exact future deployment/inspection/launch commands are in the
[persistent top-level launch section](../operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md#persistent-top-level-launch-source-gate).
The command has not run on either Pod.

## Results

No q50 behavior result exists. Host supervisor tests pass `24`; queue+consumer+supervisor tests pass
`64`. Tokenless deadline expiry cannot execute; delayed post-token rehash, a 1.15-second
acknowledgment atomic-publication stall and delayed post-ack exec all reject restart and later
converge without a fatal-before-later-runner sequence. Terminal validation freezes bytes/SHA and rejects an A-to-B
replacement. Both runtime contracts remain `prepared_not_started`, `jobs_started=0`,
`auto_start=false`.

Three additional post-link failures cover token directory fsync plus unreadable evidence stat,
token temporary cleanup and parent observation publication; all return committed pending with no
retry authority and later inspect as exact running.

## Limitations and claims not made

The host is macOS and has no Linux procfs. A Linux fake-runner smoke and fresh review are still
required before deployment. The supervisor cannot guarantee survival if an external manager
destroys the whole container/cgroup, and it does not repair pre-existing evaluation-tool closure or
hash-check-to-open TOCTOU. Nothing here is a MuJoCo score, Gate3/Gate3B result, deployment result, or
real-robot permission.

## Decision and next action

Keep the experiment preregistered and dormant. Merge only after fresh review finds no P1, then run
the no-judge Linux fake-runner smoke. Only after both gates pass may the exact one-shot `launch` be
considered; any ambiguous SSH response must be followed by `inspect`, never a second launch.
