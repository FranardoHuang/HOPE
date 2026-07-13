# Fresh SZ model-4000 four-seed matched q50

Status: completed; Linux smoke passed; both Pod results and aggregate terminal-validated

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
The exact commands were later run once per Pod through the reviewed persistent supervisor. The
runtime evidence and result hashes are recorded below and in the operation document.

## Results

Host supervisor tests pass `24`; queue+consumer+supervisor tests pass `64`. Tokenless deadline
expiry cannot execute; delayed post-token rehash, a 1.15-second
acknowledgment atomic-publication stall and delayed post-ack exec all reject restart and later
converge without a fatal-before-later-runner sequence. Terminal validation freezes bytes/SHA and rejects an A-to-B
replacement.

Three additional post-link failures cover token directory fsync plus unreadable evidence stat,
token temporary cleanup and parent observation publication; all return committed pending with no
retry authority and later inspect as exact running.

The required Linux `/proc` smoke then started only a four-second fake runner. Immediate inspection
proved exact live PID/PGID, executable, argv and environment identity. Its later
`committed_child_failed` is the expected negative result because the fake runner deliberately wrote
no Pod result; it started no judge, Kit process or simulator.

The formal launch used the same supervisor source/config SHA on both Pods. Pod1 supervisor
PID=PGID `1705148` owned seed1 then seed3; Pod2 PID=PGID `302176` owned seed2 then seed4. Both later
inspected as `terminal_result_validated`, with no remaining supervisor, child judge or q50 Kit-lock
holder. The immutable Pod results are:

- Pod1 file/content SHA `02d0e58d...645d` / `7bb91fd0...f238`;
- Pod2 file/content SHA `d31323a6...4e6f` / `eafd7b20...1899`.

One formal aggregate was then produced on Pod2 after relaying Pod1's exact result bytes into a new
no-clobber control path. Aggregate file/content SHA are
`1ba88e39e8395b8edce9365475404eafa660d4bf1b61d640a21d1d7cbb75d195` /
`226e6050c3789ebbc3145d84ca40225ab0fe9e1b868143de8ea80ad5caab648d`.
Independent canonical-JSON recomputation and fixed-value assertions passed.

| seed | parsed aggregate | parsed forehand | parsed backhand | physical root falls |
| --- | ---: | ---: | ---: | ---: |
| 1 | 50/100 | 0/50 | 50/50 | 0 |
| 2 | 88/100 | 38/50 | 50/50 | 0 |
| 3 | 98/100 | 48/50 | 50/50 | 0 |
| 4 | 0/100 | 0/50 | 0/50 | 21 |

The unchanged gate observes median `.69 < .75`, worst seed `.00 < .65`, spread
`.98 > .20`, and minimum side `.00 < .50`; all four checks fail. Seed4 is therefore
`persistent_weakness_through_model4000`, not delayed learning. This particular balance failure is
seed4-specific; the other three checkpoints recorded zero physical root falls.

The signed-face diagnostic independently blocks promotion of the apparently high seed2/3 parsed
scores. Their forehand raw-A normal error is `172.33°/174.35°`, and signed position+velocity+normal
composite success is `0/50` for both despite parsed returns `38/50` and `48/50`. Seed1 shows the same
direction (`164.86°`, signed `0/50`). Thus the old orient-normal parser is evidence of a blind
instrument, not a baseline selector.

## Limitations and claims not made

The supervisor cannot guarantee survival if an external manager
destroys the whole container/cgroup, and it does not repair pre-existing evaluation-tool closure or
hash-check-to-open TOCTOU. The result is a Python BankExam MuJoCo diagnostic, not physical-ball
truth, vendor Gate3/Gate3B, deployment evidence or real-robot permission.

## Decision and next action

Close this matched checkpoint experiment as a failed stability gate. Do not expand or promote the
`SZ` family from these parsed scores. The next measurement action is the signed-face `n/-n`
negative control, scorer correction and same-paper rerun; final behavior still requires vendor
Gate3/Gate3B. The aggregate's frozen `continue_all_arms_unmodified` action expresses that this q50
contract has no process-signal authority; it does not undo the separate human-owner resource
decision that had already stopped seed1/2/4 trainers.

Checked-in immutable evidence:

- `configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod1_result_20260713.json`;
- `configs/phase1_fresh_SZ_model4000_seed_stability_q50_pod2_result_20260713.json`;
- `configs/phase1_fresh_SZ_model4000_seed_stability_q50_aggregate_result_20260713.json`.
