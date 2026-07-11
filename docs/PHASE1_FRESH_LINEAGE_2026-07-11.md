# Phase-1 Fresh Lineage And Causal Continuations (2026-07-11)

Status: the full 179-D exact construction smoke passed. Four matched causal continuations and two
independent fresh 17,000-iteration seeds are live across all six GPUs at fixed training checkout
`6d93bcb16c422a2f42748c2dc99432559653480b`. Every arm reached its first PPO iteration and wrote a
checkpoint whose embedded schema/SHA/lineage flag matches its adjacent hard contract. Training and
formal evaluation are not complete, so this is not a claim that G05 or G06 is done.

## Scope And Lane Separation

This work deliberately keeps two lineages separate:

1. **Fresh formal lane:** runtime-order schema-2 motion, a same-family schema-v3 train/exam bank,
   an instantiated zero-friction plant, and a new checkpoint trained from scratch. No legacy
   checkpoint or legacy motion escape is permitted.
2. **Legacy causal lane:** two controlled old-vs-fixed face-pairing comparisons resumed from the
   historical M3c and M2f checkpoints. These runs use their original motion families and recovered
   train banks. They are diagnostic and permanently `evaluation_contract_exact=false`.

Nothing in this ledger authorizes a real-robot command test.

## Fresh Motion Migration Incident And Recovery

The first link-origin-to-COM migration incorrectly used
`/workspace/franco/body_order_isaac.txt` (SHA-256
`36d7a81b30a7191622db8a944040d4e8f23585c30293add9622ac0efe4c6c39c`) as both the
source and runtime body order. Its hip yaw/roll columns disagree with the live 32-body Isaac
articulation at zero-based indices `4, 5, 7, 8`.

A real Kit construction smoke failed closed with `body_names/order does not match the runtime
articulation`. The preserved log is
`/workspace/codexschema/phase1_fresh_20260711/smokes/zero_friction_contract/run.log`, SHA-256
`2c5cc82825a4024cb0204c713879d867ec898e95e39c1c2c5b506d8d3e46b7d4`. The invalid motions and
their derived banks remain quarantined under
`/workspace/codexschema/phase1_fresh_20260711/assets/v4rg/`; their immutable warning record is
`OBSOLETE_DO_NOT_USE.json`, SHA-256
`59ce1bcbe60e3ee594f3ccd4061690ee324de963b8fe84208a463190e03a3388`. They must not be copied,
trained, resumed, or relabeled as fresh. The failed Kit process was cleaned up by exact process
group `1301251` (`TERM`, then `KILL` after five seconds), not by a broad process-name kill.

The migration tool was then extended with an explicit target body order. It now reorders all four
body-indexed arrays (`body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`) from the
declared source order into the runtime order before converting link-origin velocity to COM
velocity. The tracked runtime order is `configs/a3_runtime_body_order.txt`, SHA-256
`1cdae4ba7c8d604428ee69ed4a3059e67fb195b22e1d0e294d509c4325809a3a`.

The first corrected outputs were written under `assets/v4rg_runtime_order_v2/` with diagnostic
filename suffixes. Their bytes were valid, but the bank generator rejected the renamed basenames
because they were absent from the annotation/baked-grip registry. That was the intended fail-closed
behavior; no v2 bank was accepted. The same verified motion bytes were copied with their registered
basenames into the v3 directory before bank generation.

## Audited Fresh V3 Assets

Canonical ignored root on Pod 1:

```text
/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/
```

The tracked, reviewable source of truth is
[`configs/phase1_fresh_v3_asset_manifest_20260711.json`](../configs/phase1_fresh_v3_asset_manifest_20260711.json),
SHA-256 `0c2a565d7b7040afdda97baecdaf2cea923beaf3cf9c45a574d218bb82386e46`. It binds the source motion
hashes, frame counts, strike phases, body-order hash, bank counts, physics hashes, tool hashes, and
validation results below. All four remote NPZ files (two motions and two banks) are mode `0444`.

| Artifact | SHA-256 | Audit result |
| --- | --- | --- |
| `hope_forehand_v4rg_cal.npz` | `f2cb2d9f5d27cefbcee0b790000fcd979abaf02894d4fcad061ebca27f141687` | schema 2, 50 Hz, runtime body order |
| `hope_backhand_v4rg_cal.npz` | `1722553375cd28f9b2d567c01b1a5fc6bcd149fa12cadb20e5202a9153367534` | schema 2, 50 Hz, runtime body order |
| `s1_v4rg_runtime_order_schema3_train.npz` | `2da2bd1280c45944418d41fe5788d09d7c0ebb0ff7d34fa87c8dd0fcf16a0700` | FH `757/810`, BH `724/794` kept |
| `s1_v4rg_runtime_order_schema3_exam.npz` | `d7db2568beee990ef1d64b2dce9f0ab56ca76377f8993d820b6388292d0f5096` | FH `183/190`, BH `188/206` kept |

The train and exam banks share source-family SHA-256
`b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5` and have disjoint content
IDs. The bound physics source SHA-256 is
`2e58221442665ddad7cc6dcc18d5c811dec1b0c47439b81c1c744b5148169a27`; its physics-contract
SHA-256 is `70242d798f5b97e1405df7dedfd22a5f81421c9c03127e71c254982236cfad35`.

## Corrected Runtime-Order Smoke

The motion-only Kit smoke passed with the corrected bytes:

- run log:
  `/workspace/codexschema/phase1_fresh_20260711/smokes/runtime_order_v2_motion/run.log`, SHA-256
  `fe777634c461e3adb57fbf9369ba442f3c9c84cc87f71d72e08ebd561568d6b0`;
- instantiated hard contract:
  `/workspace/codexschema/nohope_plant_test/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-11_00-07-16_phase1_runtime_order_v2_motion_smoke/params/training_contract.json`,
  SHA-256 `a21808166a1a0f4fbd77b99207c92d5e573194b68bf5e6684972c80bd6381ab9`;
- `motion_kinematics_exact=true` for both clips;
- `ZERO_FRICTION_RUNTIME_OK`: all `31/31` instantiated PhysX friction coefficients were exactly
  `0.0`.

This smoke used the 175-D deploy-parity actor and no question bank. It proved the corrected motion
order and instantiated plant checks only.

The subsequent full construction gate passed at checkout `6d93bcb`:

- launcher evidence:
  `/workspace/codexschema/phase1_fresh_20260711/runs/phase1_fresh_v3_179_smoke/`;
- run log SHA-256:
  `684ee92ca2abf7f1b9321f0842dd2df8eb1dbd4621f7afa5d9ce80eac2806092`;
- run directory:
  `logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-11_00-48-24_phase1_fresh_v3_179_smoke`;
- hard-contract SHA-256:
  `3a3b3d956e19d47f7e6f0a157159dc96c8f09d8345c436a776c8c7e99c0b9972`.

The process exited cleanly after `max_iterations=0`. Independent formal validation confirmed the
179-D `deploy_parity_face179` actor, `face_command_enabled=true`, `shared_plus_y`, both exact
schema-2 motion contracts in the live articulation order, exact schema-v3 train bank SHA
`2da2bd...a0700`/family `b21c161a...28ad5`, `motion_kinematics_exact=true`, the legacy-motion
switch off, and exactly 31 zero PhysX friction coefficients.

## Recovered Legacy Causal Banks

The causal continuations must use train banks reconstructed from the exact historical family. They
must not borrow the fresh v3 bank.

| Family | Train bank SHA-256 | Source family | Kept | Historical exam reference |
| --- | --- | --- | --- | --- |
| v4rg | `dc67326f5cf0e1a3e3a6ae89f43cbd8a9a3785c341ab2ae998e3edbd40f8d30a` | `d093246b4a6aab81d6db50164515c989e2da1435a1e03b046272645dd9e53d9f` | FH `757`, BH `724` | `10917148ef251a4dabe387ea418b8907c26fe320b95d9ff874380d09f73e5bb2` |
| swing | `96329c79f13e659c035bf65bafedf84123d23a3219b02ac78ae654aed930cf60` | `3344b610e230bc55a0eca5647cfb55f338a7635bd4bb4025a579fc347b05f3d7` | FH `803`, BH `765` | `750f1df4ebaf851b96495c53ebbd083c06275f254ce19862f2e84183ec45cb0e` |

The v4rg and swing manifest SHAs are respectively
`6ff25a94c2a5abb590e84415458a8958f749bb62f34f0890d7b7922891571a74` and
`04751e0424458a85f033519ee033176169f5479013abe5a85a46bc1488970fbe`. Both manifests record
Torch closed-loop success, train/exam content-ID disjointness, and byte-identical recovery of the
historical exam-generation parameters. The bundle-level `BUILD_RECORD.json` SHA-256 is
`95b9727489d1aa298e411e1ec589505b2c3da1720569fa0da33ddd10176c6032`.

These banks remain legacy-motion family artifacts. Continuations must set the explicit legacy
motion diagnostic override and preserve `evaluation_contract_exact=false`.
Read-only copies of the references now sit beside their train banks as
`s1_swing_schema3_exam.npz` and `s1_v4rg_schema3_exam.npz`, with the table hashes above, so
`judge.sh` can derive `train -> exam` without borrowing another family or requiring an ambient
historical path.

## Four-Arm Causal Design

Within each motion family, the only intended scientific variable is
`task.racket.face_command_pairing`:

| Arm | Resume checkpoint | Motion / train bank | Pairing | Live evidence |
| --- | --- | --- | --- | --- |
| M3-old | M3c `model_16999.pt`, SHA `46f0050589f3343d96f2e5c261b92224079b379e4da473be342b1bd0f0cf7ff1` | legacy swing / `96329c...cf60` | `legacy_signed_vs_A` | Pod 1 PGID `1310472`; run `00-50-50`; contract `7542c59b...d941b` |
| M3-S1 | same M3c checkpoint | same legacy swing bank | `shared_plus_y` | Pod 1 PGID `1311109`; run `00-51-29`; contract `d3ff715e...9d9ce` |
| M2-old | M2f `model_16999.pt`, SHA `0ab05144ec1792db91d6e1e3c2ce79f46dae9507ed267b1470838bf998f0f012` | legacy v4rg / `dc6732...8d30a` | `legacy_signed_vs_A` | Pod 2 PGID `161096`; run `00-50-51`; contract `75af3b9d...a99dc` |
| M2-S1 | same M2f checkpoint | same legacy v4rg bank | `shared_plus_y` | Pod 2 PGID `162194`; run `00-53-31`; contract `7268eb38...28f2` |

The M3c and M2f checkpoint paths are recorded in
[setup_local_sync.md](operations/setup_local_sync.md). Each pair keeps its checkpoint, motion,
train bank, seed (`1`), environment count (`4096`), additional iteration budget (`4000`), plant,
timing, noise, target sampler, reward scales, and face-guidance value fixed. M3 keeps its inherited
generic racket guidance `-0.95`; M2 keeps `0.0`. Both use the 179-D face-command actor contract and
remain inexact because the warm start and motion inputs are legacy.

Before launching an arm, a construction smoke must show the resolved 179-D term order, hard
contract, correct train-bank family, resume checkpoint, and first learning iteration. Kit boots are
serialized per Pod; a failed run is stopped by its exact PGID. Do not add the proposed S1-only face
guidance (`-0.4`, `theta=pi`) unless the paired S1 result still exhibits the face dead zone; that is
a later intervention, not part of this four-arm comparison.

The executable manifest is `scripts/launch_phase1_20260711.sh`. It hashes every input, pins the
shared recipe and per-family deltas, and uses `scripts/launch_kit_training_locked.sh` so only Kit
boot is serialized; training jobs run concurrently after their first iteration marker. Its Pod
allocation is M3 old/S1 + fresh seed 1 on Pod 1 and M2 old/S1 + fresh seed 2 on Pod 2. The fresh
seeds are independent 17,000-iteration, 4,096-environment runs from scratch. Use
`PHASE1_DRY_RUN=1` for a no-Kit command/asset audit.

Pre-launch export review closed three post-training traps. Diagnostic schema-3 continuations now
export only when their complete structural sidecar and checkpoint SHA binding agree, and remain
`training_contract_exact=0`; an exact-lineage claim still requires exact schema-2 motion. Judge
restores zero friction and the 175/179/181 actor layout from the checkpoint-adjacent hard contract
rather than task defaults. Finally, `face_command_enabled` joins `face_command_pairing` in the hard
contract and ONNX metadata, so a same-width reward/critic semantic change cannot pass as a resume.
The focused contract/judge subset now passes `38` tests.

The first invocation of the full smoke stopped during Hydra composition, before Kit or a GPU was
created: the one-off launcher used undeclared `task.decimation=4`. The declared composed path is
`task.sim.decimation=4`; the launcher was corrected and the failed arm state retained as a
pre-construction record rather than counted as a runtime smoke.

## Six-GPU Launch Audit

Both Pods passed the final source tests at the launch commit: the Isaac/Hydra group passed `130`
tests and the full dependency-supported union passed `153`. The three GPU lanes on each host then
reached a real `Learning iteration` marker. Their first saved checkpoints prove the intended
lineage binding:

| Arm | First checkpoint | Embedded contract binding | Lineage exact |
| --- | --- | --- | --- |
| M3 old / S1 | each `model_17000.pt` | schema 3; SHA matches the corresponding sidecar | `0` |
| M2 old / S1 | each `model_17000.pt` | schema 3; SHA matches the corresponding sidecar | `0` |
| fresh seed 1 / 2 | each `model_0.pt` | schema 3; SHA matches `3a3b3d...b9972` | `1` |

The two M3 hard contracts differ only at `face_command_pairing`; the two M2 contracts also differ
only at that key. This is a recursive JSON comparison, not a launch-command assumption. Fresh
seed 1 runs on Pod 1 PGID `1311754` (`00-52-16`) and seed 2 on Pod 2 PGID `162836`
(`00-54-53`); both use contract SHA `3a3b3d...b9972`, zero friction and exact motion/bank facts.

One Pod 2 M2-S1 boot aborted at scene creation with `malloc(): invalid size (unsorted)` before a
hard contract or PPO iteration. PGID `161672` exited by itself, GPU 1 was empty, and host memory
was not exhausted. The preserved failed directory is
`runs/phase1_M2_S1_pairing_malloc_abort_1/` (log SHA `c0e3f71c...b90a8`, launch-state SHA
`d72908c1...544f`). An unchanged-command single-arm retry reached the first iteration and is the
only accepted M2-S1 run.

Read-only `judge.sh --dry-run` checks on M3-old, M2-S1 and fresh seed 1 resolve the correct motion
pair, phase pair, adjacent exam bank, 179-D actor and plant replay. Diagnostic runs receive the
explicit inexact evaluator escape; the fresh exact candidate does not. The judge environment also
forces the current checkout's `setup_train_env.sh`/source-first Python path rather than inheriting
another user's checkout.

## Remaining Gates

1. Monitor all six immutable-checkout runs to terminal checkpoints; do not fast-forward either Pod
   checkout while a local arm is alive.
2. Verify terminal checkpoint iteration, sidecar SHA binding, lineage flag and finite parameters.
3. Export every causal terminal as diagnostic and both fresh terminals as exact candidates, then
   run the same immutable exam schedule in Isaac and MuJoCo. A legacy resume can never be promoted.
4. Add completion/result hashes here and to G05/G06; only then decide whether the S1 guidance arm
   is warranted.
