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

The post-launch judge correction was verified separately at `08e438e` in a detached evaluation
worktree (`154 passed`); the live training checkout remains untouched at `6d93bcb`.

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

## Capacity And Checkpoint-Curve Correction

Occupying all six GPUs with one process each was not the established meaning of "full". The measured
breadth rule is four 4096-env jobs per GPU (about 22--23 GiB on a 32.6-GiB 5090), 75-second serialized
Kit boots, and checkpoint exams during training. The target is therefore
`4/4/4 + 4/4/4 = 24` live arms, not six.

The pre-registered scale-out matrix is
[`configs/phase1_scaleout_matrix_20260711.json`](../configs/phase1_scaleout_matrix_20260711.json).
It uses a second paired continuation seed for M3/M2 and a four-seed fresh 2x2 factorial over
face pairing and plant friction. `SZ` (`shared_plus_y`, zero friction) is the formal target cell;
the other fresh cells are causal diagnostics and cannot silently replace it. Guidance, N1, R8 and
later-stage variables are not mixed into this matrix.

Both Isaac and MuJoCo treat the explicit diagnostic escape as a one-way downgrade: it authorizes
the run and forces `evaluation_contract_exact=false`. Thus `LZ/LP` cannot inherit a bookable label
from otherwise exact fresh motion/checkpoint provenance. `SP` is also explicitly inexact: its
non-zero PhysX coefficient has no exact MuJoCo `frictionloss` equivalent, even though its face
pairing is shared. Only `SZ` is the formal target cell. Generated SP/LZ/LP jobs pass the diagnostic
escape explicitly so an inexact queue-head job cannot fail the formal profile and block later SZ.

The missing early curve is also being repaired. Causal `17000/18000/19000` and fresh `0/1000/2000`
checkpoints are scheduled through the detached evaluator first; subsequent milestones follow the
1000--2000 iteration policy. The first attempt exposed two evaluator preflight faults, not model
failures: the detached worktrees lacked the ignored A3 asset link, then `play.py` wrote ONNX but its
redirected success line was lost to stdout buffering. The next preflight reached sidecar generation
and exposed four valid constant observation features with saved `_std=0`; the writer incorrectly
required `std>0` even though saved inference uses `(x-mean)/(std+0.01)`. Failed batches are retained;
the asset links are now bound to the frozen training checkout, `judge.sh` forces unbuffered export
output, and the writer accepts only finite `std>=0` with a strictly positive `std+eps` divisor.
Both Pods reproduced exactly four protected zero dimensions and rejected negative scales in tests.
One Isaac export runs per Pod at a time, shares the training Kit boot lock, and completed exports run
MuJoCo BankExam on CPU with one BLAS/OpenMP thread. Worker state binds a clean evaluator commit,
judge SHA, frozen training commit and checkpoint SHA. Full rationale, early-stop protection and peak-density rules are in
[`phase1_ablation_acceleration_2026-07-11.md`](research/phase1_ablation_acceleration_2026-07-11.md).

Two later preflights were also evaluator failures and are retained separately. The first reached
MuJoCo only after ONNX and both sidecars succeeded, then found that
`/workspace/hope_mjeval_venv` had `onnxruntime` without the `onnx` graph package. The corresponding
workers/judges alone were stopped by their recorded PGIDs; both training checkouts and all six
trainers were untouched. The archives are `initial_pod{1,2}_missing_onnx_pkg_4`. Both Pods now pin
`onnx==1.22.0` (`ml_dtypes==0.5.4`), and `onnx.checker` plus ONNX Runtime accepted inputs
`obs [1,179]` and `time_step [1,1]`.

The next retry exposed a formal armature threshold below serialization precision. The same decimal
A3 plant differs by at most `2.71e-9` after Isaac's float32 metadata path versus MuJoCo's float64
XML path. A preliminary `1e-8` armature bound passed that field, then exact retry exposed the same
issue at the `118.2` ankle effort limit (`118.199996948...` in training metadata; max difference
`3.0517578e-6`). The final comparator uses exact float32-grid identity rather than a larger fixed
tolerance: exact float64 matches pass, otherwise canonical float32 metadata and MJCF must map to
the same grid point. Fresh `0/1000/2000` rows from these attempts are pre-rollout records, not
model scores. The causal rows completed because their historical plant is an explicit inexact
diagnostic; their summary JSON is authoritative. A report-layer defect that printed the bank leg's
exactness instead of final artifact exactness in `DENOMINATORS` is fixed so legacy reports also say
`evaluation_contract_exact=false`.

The first implementation of that report-only correction touched
`venue_ball_sampler.py`; formal export correctly rejected it because that module's complete bytes
belong to the immutable bank physics hash. The failed clean-q10 retry was stopped only by its
recorded judge PGIDs after the Isaac failure path hung during shutdown, then retained as
`fresh_retry_pod{1,2}_clean_q10_de13800`. The sampler was restored byte-for-byte to SHA-256
`00e28e85...30cc`; the final exactness rewrite now lives in `mujoco_eval_onnx.py`, outside the bank
physics source set. This is another evaluator preflight, not a model score.

The corrected fresh retry is deliberately cheaper than the historical repair:
`configs/phase1_checkpoint_curve_fresh_retry_pod{1,2}_20260711.json` freezes
`ns=0`, `K=20` (10 questions per side) for `0/1000/2000`. It measures growth
direction only. No arm may be stopped or promoted from that small paper; those
decisions require the pre-registered 50-per-side clean schedule, and robustness
noise is run only after a candidate survives.

## Full-Pool Scale-Out Evidence

All three scale-out layers are now accepted. Each of the six RTX 5090s ran four 4096-env trainers
concurrently. Pod 1 used `23.24/23.06/23.17 GiB` on GPU 0/1/2 at `93/93/94%`; Pod 2 used
`23.06/22.87/22.87 GiB` at `87/96/97%`. Host available memory was `840 GiB` and `904 GiB`.
Every one of the 24 pre-registered experiments reached its first PPO iteration, wrote a finite
first checkpoint, and matched the embedded `training_contract_sha256` to its adjacent contract.
Fresh contract SHAs collapse exactly by cell as expected (for example `LZ=0f65930c...bb06` and
`LP=b9feb4d5...123e`); `LZ/LP` training lineage remains structurally exact but their legacy pairing
forces evaluation inexact.

Pod 1's first `phase1_fresh_v3_LZ_seed3` boot aborted before a contract/iteration with
`malloc(): invalid size (unsorted)`. Its preserved run log and launch-state SHAs are
`d66a8043...951d` and `0f004c18...b768` under
`phase1_fresh_v3_LZ_seed3_malloc_abort_1/`. No process survived and no host-memory pressure was
present. The identical-command single-arm retry reached ready as PGID `1354525` and is the only
accepted LZ seed 3. No broad process signal or training-checkout change was used.

## First Exact Growth Curve And Ongoing Cadence

The final float32-grid plant comparator cleared a single fresh `model_0` probe and then both
pre-registered clean q10 retries completed at evaluator commit `c711a03`. All six fresh jobs
(`seed 1/2 x 0/1000/2000`) returned `rc=0`, used the same exact schema-v3 exam schedule
(`K=20`, 10 per side, seed 0), and emitted `evaluation_contract_exact=true`.

The all-side return-rate direction was:

| Fresh SZ seed | model 0 | model 1000 | model 2000 |
| --- | ---: | ---: | ---: |
| 1 | 0.00 | 0.50 | 0.90 |
| 2 | 0.00 | 0.50 | 1.00 |

At 2000, seed 1 was FH/BH `0.80/1.00` and seed 2 was `1.00/1.00`. This is the first successful
formal checkpoint growth curve; it demonstrates useful growth well before terminal. It is still a
small direction screen and cannot stop or promote an arm without the separately frozen q50 paper.

Seed 1 then supplied the first direct example of why the paper must be run before terminal. Its
same-schedule q10 fell from `0.90` at model 2000 to `0.50` at model 4000, triggering the separately
preregistered exact q50 without stopping the trainer. On one clean K=100 schedule (semantic SHA
`7dc6af82...ff3e`, 50 per side), model 2000 returned FH/BH/aggregate
`33/50,50/50,83/100`; model 4000 returned `0/50,50/50,50/100`. Both results are fresh lineage,
`evaluation_contract_exact=true`, finite and bound to the same schema-3 contract
`3a3b3d95...b9972`. Model 2000 is therefore the retained checkpoint inside this frozen pair, while
the whole seed-1 arm continues unmodified. All 100 questions in both cells finalized through the
evaluator's non-physical post-strike tracking guard (`guard_reset=true`, `physical_fall=false`), so
this isolated one-question paper is not continuity or deploy-stability evidence. The complete
binding is `configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json`.

The byte-identical Isaac companion has also completed under fresh/exact
semantics. Both model 2000 and model 4000 scored FH/BH/aggregate
`49/50,50/50,99/100`, with 99 exact reaches, one guard reset and zero physical
falls. Isaac therefore gives delta zero and does not reproduce MuJoCo's
`+0.33` model-2000 advantage. The preregistered final tie-break names the
earlier checkpoint only after a complete Isaac tie; that is not independent
cross-engine support. Model 2000 remains the MuJoCo-selected checkpoint inside
this frozen pair, the arm continues, and the cross-engine/formal deployment
ranking gate stays open. Full runtime/scorecard/ledger hashes are in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_isaac_result_20260711.json`.

The original causal arms also completed clean q10 at 20000. M3-S1 scored `1.00` versus M3-old
`0.45`; M2-S1 and M2-old both scored `0.50` on this small prefix. All four remain diagnostic with
`evaluation_contract_exact=false`. The long-lived original-arm workers then advanced to wait for
fresh 4000 and causal terminal points.

The additional 18 scale-out arms are now covered by deterministic manifests generated from their
actual run directories. Four queues split causal/fresh per Pod: causal seed 2 at
`18000/19000/20000/20998`, fresh at `2000/4000/.../16000/16999`. They cover 18 unique arms and
142 clean q10 jobs. Milestone-major ordering keeps like-age comparisons together, while separate
causal/fresh queues prevent a continuation terminal file from blocking an earlier fresh screen.
The binding spec marks every job `screen_only`; q50 remains mandatory for a decision.

The causal terminal index was corrected from the preregistered-but-impossible
`20999` to the runner's actual final saved iteration `20998`. Read-only Pod2
validation proved M2-S1 exited normally at `20998/20999`, checkpoint field
`iter=20998`, with 1,762,715 floating elements all finite and the embedded
contract SHA matching the adjacent schema-3 sidecar. The original cadence
worker had been waiting for `model_20999.pt` forever. Only the exact cadence
and causal-worker PGIDs were stopped; corrected runtime manifests now wait for
`model_20998.pt`, while fresh workers and all trainers were untouched. This is
a scheduler correction, not a recipe or evaluation-paper change.

The full machine-readable audit is
`configs/phase1_M2_S1_terminal_audit_20260711.json`. Reproduce its checkpoint,
finite-tensor and adjacent-contract assertions on Pod2 with:

```bash
RUN=/workspace/codexschema/nohope/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-11_00-53-31_phase1_M2_S1_pairing \
/workspace/hope_isaac_venv/bin/python - <<'PY'
import hashlib, json, os
from collections.abc import Mapping
from pathlib import Path
import torch

run = Path(os.environ["RUN"])
checkpoint_path = run / "model_20998.pt"
contract_path = run / "params" / "training_contract.json"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

def tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)

all_tensors = list(tensors(checkpoint))
floating = [value for value in all_tensors if torch.is_floating_point(value)]
nonfinite = sum((~torch.isfinite(value)).sum().item() for value in floating)
checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
assert checkpoint["iter"] == 20998 and nonfinite == 0
assert checkpoint["infos"]["training_contract_sha256"] == contract_sha
assert not bool(checkpoint["infos"]["training_contract_lineage_exact"])
print(json.dumps({
    "checkpoint_sha256": checkpoint_sha,
    "contract_sha256": contract_sha,
    "tensor_count": len(all_tensors),
    "floating_tensor_count": len(floating),
    "floating_elements": sum(value.numel() for value in floating),
    "nonfinite": nonfinite,
}, sort_keys=True))
PY
```

The original-arm cadence is also split into independent causal and fresh
workers. `phase1_checkpoint_curve_cadence_podN_20260711.json` now contains
only the 20000/20998 causal pair. The Pod1 fresh manifest begins at 4000
because only 0/1000/2000 had preserved results when the earlier worker was
replaced; the Pod2 fresh manifest begins at 6000 because its 4000 result was
already preserved. Thus an old causal terminal cannot block either original
fresh curve. All current q10
manifests set top-level `screen_only=true`, `stop_or_promote_allowed=false`
and repeat `screen_only=true` per job; the checked-in worker rejects an omitted
or contradictory policy, requires each `--schedule-k` to equal the top-level
q10 contract, records the complete manifest SHA, and binds the canonical
screen-policy-plus-job contract SHA into completed state before skip/reuse.
This guard does not turn q10
into a decision paper; q50 remains separate.

Pod2's corrected terminal q10 pair completed before the split: M2-old returned
FH/BH `0/10, 8/10` (aggregate `0.40`), while M2-S1 returned `0/10, 7/10`
(`0.35`) on schedule SHA `75aca567...51d7`. Both exports/judges returned zero
and both remain `evaluation_contract_exact=false`. This small terminal prefix
does not reproduce an S1 gain and both forehands are zero, but it is explicitly
non-decisive; q50 is required before stopping or selecting. Full checkpoint,
report and summary hashes are in
`configs/phase1_M2_terminal_q10_pair_20260711.json`; the M2-S1 terminal audit
JSON has also been updated from unjudged to this preserved result.

Pod1's M3 terminal pair is now closed to the same integrity level. M3-old
`model_20998.pt` is finite, has SHA `320b77c9...417a`, embeds the same
`7542c59b...d941b` SHA as its adjacent contract, and correctly retains
lineage exact=`0`. The immutable terminal q10 schedule SHA is
`7a908142...d614`; M3-old FH/BH/aggregate is `0.50/0.40/0.45`, while M3-S1
is `1.00/1.00/1.00`, for paired aggregate delta `+0.55`. The machine-readable
audit/result are `configs/phase1_M3_old_terminal_audit_20260711.json` and
`configs/phase1_M3_terminal_q10_pair_20260711.json`. This q10 is
direction-only and triggered, but did not replace, the preregistered shared
K=100 q50 paper.

The MuJoCo q50 pair has now completed on one materialized schedule artifact:
file SHA `69f73458...7f25`, semantic schedule SHA `949eb196...8fc0`, seed 0,
50 attempts per side and zero censored rows. M3-old returned
FH/BH/aggregate `31/50,11/50,42/100`, contact `89/100`. The original paired
summary mislabeled the evaluator's `fell=9` union as nine physical falls; the
raw ledger contains **one physical fall plus eight non-physical guard resets**.
M3-S1 returned `50/50,50/50,100/100`, contact `100/100`, with zero such
terminations. The `+0.58` aggregate delta selects M3-S1 only within this same-family
terminal causal paper. Both checkpoint lineages and evaluator contracts remain
inexact, and the result cannot authorize formal/deployment/hardware use. The
first v1 execution judged only M3-old then fail-closed on a runner assumption
about the real summary shape; v2 corrected only the validator, reproduced
identical schedule bytes and ran both cells. The preserved failed-attempt and
accepted paired-result hashes are in
`configs/phase1_M3_terminal_q50_result_20260711.json`. A same-artifact Isaac
companion has also completed. On the identical 100-question order, both M3-old and M3-S1 scored
FH/BH/aggregate `0.98/1.00/0.99`, with 99 exact reaches, one guard reset and zero physical falls.
Isaac therefore gives delta `0.00`, not MuJoCo's `+0.58`. The engines disagree on this causal
legacy comparison: S1 may be selected only inside the MuJoCo evaluator/family, and no cross-engine,
formal or deployment gate closes. Full hashes and three preserved fail-closed wrapper attempts are
in `configs/phase1_M3_terminal_q50_isaac_result_20260711.json`.

A question-aligned forensic pass now explains why neither Isaac companion can
close a cross-engine gate. “Same question order” was not “same outcome
instrument”: current Isaac scored a virtual analytic ball while MuJoCo scored
ONNX execution plus physical contact/flight. For fresh SZ model 4000, mean FH
racket-center error is `13.15 cm` in MuJoCo (all 50 exceed the frozen `9.5 cm`
capture margin) but only `2.48 cm` in Isaac; model 2000 is `9.03/3.03 cm`.
For M3-old BH, Isaac's analytic `orient_normal` erases the signed-face error:
mean pre-orient normal error is `168.15 deg` versus M3-S1's `3.66 deg`, yet both
receive `50/50`. The content-addressed forensic result SHA is
`aff8f4e665d20bb76a56e079735f32b6766388ee05f61c51e93adeb568be45c9`.
The follow-up 2x2 preregistration (Isaac/MuJoCo x physical/analytic) freezes the
question order and thresholds and fails closed unless all four instrument cells
exist; prereg SHA is `dd8fb0b93d2a809a2875682e9399717f856ac891a474ab214364b617176e6818`.
It is runtime-blocked because current Isaac PhysicalBall is incoming-flight
Phase A only, with no racket impulse or post-contact physical truth.

The four causal-triangle refill workers initially used eval `46a0ce2`'s
legacy state schema. Their commands/results were valid but omitted manifest,
job-spec and job-contract SHAs. A Pod-atomic correction verified both workers
on each Pod as exact-PGID, single-member and childless, preserved their old
state/logs, TERM-signalled only PGIDs `1410648/1412047` and
`196753/197939`, then started standalone hardened worker `21e30153...` in
fresh state directories. Current PGIDs are Pod1 `1416771/1416784` and Pod2
`198759/198771`. All four rejudged 17k jobs returned rc=0 and now bind the
manifest/job/job-contract/checkpoint/judge and clean train/eval commits; the
correction-sidecar SHAs are `2faf88de...ffe3`, `1d6f8ba3...bae9`,
`0dd02fae...d165`, and `45f4334d...0ad`.

The six older global cadence workers were then hardened under a separate exact-PGID transaction.
Only legacy worker PGIDs Pod1 `1394810/1380340/1397266` and Pod2
`194276/192815/195085` received TERM; no trainer, judge or checkout was changed. Their replacements
are Pod1 `1432280/1432292/1432304` and Pod2 `200706/200718/200730`, all running worker
`21e30153...`. Five already-available legacy results were not trusted for hard reuse and were
rejudged rc=0 under manifests that bind manifest/job/job-contract SHAs; causal commands explicitly
retain the inexact escape. Attestation, transaction, correction, launch and rejudged-state hashes
are in `configs/phase1_global_curve_worker_hardening_result_20260711.json`.

Queue discipline is now checked before runtime as well as documented. Run
`python3 scripts/validate_phase1_queue_governance.py` before copying or launching any curve
manifest. The validator accounts for all 142 scale-out jobs and all 24 cadence plan slots, enforces
q10 K=20/10-per-side screen-only semantics, checkpoint order and readiness barriers, and rejects
q50 work from the generic curve worker. This does not retrofit a running process, but it prevents a
tracked manifest or documented launch path from silently bypassing the preregistered discipline.

The ten new air-swing diagnostic GMR paths also completed their first physical
placement transform. Each was independently shifted in root-z using tool
`db5bd167...` and canonical MJCF `2ab1cd31...`; original discrete-frame
penetration was `8.072--8.716 cm`, and every output leaves about `10 um`
minimum ground clearance. All input/output/report and compiled-collision
bindings are in `configs/motion_video_gmr_ground_results_20260711.json`.
This does not join the Phase-1 exact lineage: per-video betas, continuous
collision, dynamics, table/net, returnability and schema-2 are still absent.

The separate canonical-body-shape branch has now advanced beyond those
per-video-beta diagnostics. Ten equal-video-weight canonical GVHMR PTs were
retargeted CPU-only through clean GMR `aabea2e`; loader SHA
`2737f472...5de2` consumes exactly ten beta components with no zero padding.
All ten canonical GMR outputs are finite 30 Hz/31-DoF artifacts with
frame-zero warm-up below `1e-4`; full hashes are in
`configs/motion_video_canonical_gmr_results_20260711.json`. Canonical grounding
then passed 10/10 by changing only fixed root-z, and the accepted v4 dense
screen found zero ground danger, self-collision, `<5 mm` racket/body danger or
`<20 mm` warning across 654 source frames and 5,162 samples at 240 Hz; minimum
body clearance is `40.2466 mm`. Returnability remains deliberately null because
the GMR-world to HOPE +X/table transform and mirror contract are not verified.
Thus these paths still do not join Phase-1 exact lineage or enter RL.

The first adjacent scale-out causal curve also demonstrates why q10 cannot be
a decision paper. On the same K20 schedule `75aca567...51d7`, M2 seed2
old/S1 aggregate changed from `.40/.60` at 18k to `.50/.40` at 19k. Both 19k
checkpoints are finite and bind iteration, adjacent contract SHA and causal
lineage. The crossing is preserved in
`configs/phase1_M2_seed2_18k_19k_q10_curve_result_20260711.json`; both arms
continue unchanged and no q50 is triggered.

## Continuous-Timing Boundary

The current runs do carry the robot state across clip wraps, but they do not establish arbitrary-time
continuous play. With the bound 141/134-frame motions, strike frames 66/45 and hold `U[0,100]`,
the theoretical same-player strike interval is about `q10/median/q90=2.90/3.75/4.60 s` and cannot
be shorter than 2.40 s. The conservative venue A-B-A audit (`n=21`) measured
`1.757/1.903/3.356 s`; the opponent's next hit appears a median 0.951 s after ours, before the
current clip-wrap installs a new task. These are overlapping windows, 16/21
come from high-ball practice and the 2.5 s leg filter right-censors the slow
tail, so `1.903 s` is a falsification datum, not a target distribution.

This is a contract boundary, not a reason to mutate the live 24-arm pool. A separate `T0/T1`
timing/carry-state pair must compare complete-clip wrap against post-strike event-driven next-task
installation, with a longer/opportunity-count episode and immutable interval schedule. The source
SHA, reproducible filter, proposed contract fields and checkpoint metrics are in
[`phase1_continuous_rally_timing_2026-07-11.md`](research/phase1_continuous_rally_timing_2026-07-11.md).
The content-addressed preregistration uses a separate balanced engineering
event grid rather than fitting that median. Its design-check passes and its
launch-check intentionally fails on missing implementation/schedule/judge/
self-hit/plant bindings; prereg SHA is
`2e7c4a344c0f2f81f67fd1246e5a724eaef92570c45e830c85f298377f52289c`.

Plant semantics form an independent deployment blocker. `SZ` is exact only for the current
zero-friction execution protocol; `SP/LP` are historical direct-number proxies and cannot estimate a
physical friction main effect. The preregistered repair requires one physical latent friction model,
separate PhysX/MuJoCo adapters and a fresh shared-face `Z/C x 2 seed` axis before any calibrated
plant claim. See `docs/research/phase1_plant_semantics_repair_2026-07-11.md` and
`configs/phase1_plant_semantics_repair_prereg_20260711.json`.

## Remaining Gates

1. Keep the 28 accepted arms (currently four terminal plus 24 live) and all
   live cadence workers under exact-PGID monitoring; do not fast-forward either
   frozen training checkout while a local arm is alive.
2. Finish the pre-registered checkpoint curves. Compare old/S1 only within the same family, seed and
   milestone; preserve peak checkpoints as well as terminal checkpoints.
3. Verify every promoted checkpoint's iteration, sidecar SHA binding, lineage flag and finite parameters.
4. Export causal checkpoints as diagnostic and the fresh `SZ` target seeds as formal candidates, then
   run the same immutable exam schedule in Isaac and MuJoCo. A legacy resume/pairing can never be promoted.
5. Add completion/result hashes here and to G05/G06; only then decide whether an S1 guidance arm or
   the separate `T0/T1` continuous-timing pair is warranted.
