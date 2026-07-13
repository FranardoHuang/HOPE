# Run The Phase-1 Cross-Engine Instrument-Parity 2x2

## Current Status

`Revoked for the current exact lane / historical artifacts only`. Do not count or launch this
paper from any checkout. The historical prerequisite contract is
`configs/phase1_cross_engine_instrument_parity_2x2_prereg_20260711.json`; its validator must
reject it as revoked. The Isaac rider is separately frozen by
`configs/phase1_isaac_bank_exam_physical_truth_phase_b_contract_20260711.json`. Source completion
never became simulator evidence, and that rider is now also rejected directly by the evaluator
loader because its checkpoint predates explicit actor leg-reference mask provenance.
The historical preregistration SHA is `bd90f6f2...0175`, and revoked Phase-B rider-contract SHA is
`1af7a0b3...b376`. The immutable
revocation receipt is
`configs/phase1_cross_engine_instrument_parity_2x2_revocation_20260713.json`.

This operation is simulator-only. It authorizes no real-robot command or deployment test and does
not change, stop, promote, or restart a training arm.

## 这张卷为什么存在

同一组 100 道题曾在两引擎里得到不同排名，但双方已有的回球列实际上都是解析判分，且引擎执行和
判分仪器没有被干净拆开。尤其是当前 Python MuJoCo BankExam 没有仿真中的球拍—球接触：它只物理
推进机器人，再把击球时的球拍状态交给 `VirtualReturnScorer`。因此“同一题序”不是物理一致性
证据。冻结的 2×2 要补齐四个真实格子，分别解释引擎执行差异和判分仪器差异：

| 引擎 | 物理真值 | 解析对照 |
| --- | --- | --- |
| Isaac | 必须用新 post-epoch rider 补；旧 Phase-B 源码/合同已撤销 | 已有解析诊断仍需标准化、内容绑定的格子输出 |
| MuJoCo | 必须补；当前 BankExam 没有物理格 | 已有 Python BankExam/`cf_*` 解析诊断仍需标准化、内容绑定的格子输出 |

任一行或列都不能替代另一格。尤其是把 Isaac 解析回球改名成“物理回球”时，校验必须直接失败。

## Historical Revoked Paper

- target: fresh exact SZ seed1 `model_2000`, checkpoint SHA
  `99e82659...ae4c`, hard-contract SHA `3a3b3d95...b9972`;
- bank SHA `d7db2568...5096`;
- schedule file SHA `66e89986...1cb3`, semantic SHA `7dc6af82...ff3e`, 100 attempts,
  50 per side, one question per reset, no wrap/no censoring;
- question-order SHA `b87e81a3...1f91`;
- analytic capture radius remains `0.095 m` and minimum approach speed remains `0.3 m/s`.

Changing a threshold to reproduce the MuJoCo ranking creates a different paper and is forbidden.

## Isaac Instrumentation Extension

The evaluator keeps the accepted `hope.isaac-bank-exam.v1` score fields intact and adds the
JSON-only `hope.cross-engine-state-instrumentation.v1` extension. It exports:

- the complete numeric ready state: environment-local root state, explicit joint order, joint
  position and joint velocity, plus both the legacy ready-state digest and a content SHA;
- one numeric state snapshot for every question, at exact strike or at termination before exact:
  base root state, racket position/velocity, target state, incoming ball velocity/spin;
- all three face lanes: signed striking-face normal **before** `orient_normal`, raw mount `+Y`,
  and the analytic oriented normal;
- the analytic capture/net/opponent/landing outcome and an explicit physical-truth capability.

The extension does not enter observations, rewards, commands, actions, resets, or scoring. The
CSV schema remains unchanged; nested numeric state lives in the scorecard JSON.

With the rider disabled, Isaac remains Phase A: it realizes incoming flight, disables robot
collision and applies no racket impulse. The evaluator emits `available=false`, capability
`incoming_flight_only_no_paddle_contact_phase_a`; the old `hit`/`returned` fields and CSV remain
the analytic virtual-v1 score.

The contract-bound Phase-B rider adds one code-authoritative blade-disc contact scan per physics
substep. It catches both slab contact and signed blade-plane crossing, snaps the ball to the
blade plane, and delegates velocity/spin change bit-for-bit to
`virtual_ball.predict_paddle_contact`. PhysX then supplies gravity while the existing deterministic
venue wrench, code-driven table bounce, net crossing and landing crossing produce physical truth.
The ball collider remains disabled so an unfitted PhysX robot/table contact cannot double-hit the
ball. Full capability is exactly
`physical_paddle_contact_and_post_contact_flight_v1`.

The formal cell fails closed if the physics callback did not register, a contacted return has no
landing by finalization, any source/contract/target SHA differs, the checkout is dirty or attached
to a branch, or any of the 100 outcomes is missing. Physical booleans live only at
`attempts[*].instrumentation.physical_truth`; legacy virtual-v1 score fields are deliberately not
renamed or overwritten.

After installing each immutable exam row, the evaluator explicitly begins one physical attempt
generation bound to that row's schedule index. Reset-time training rows and later/repeated
resamples cannot satisfy that token. An available row additionally requires the incoming ball to
have been served and the exact-strike frame to have occurred; this separates an instrument serve
failure from a real policy miss. This seam is intentionally one-question BankExam-only and is not
a reusable continuous-T1 score lane.

Consequently a pre-exact fall/guard/timeout invalidates the entire instrument-parity cell rather
than entering the normal BankExam all-attempt miss denominator. That strictness is specific to
instrument parity: all 100 rows must receive a physical serve and reach the shared strike-state
measurement before engine/instrument differences can be compared.

## Local Source Tests

These commands start no simulator and issue no robot command:

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_isaac_bank_exam_adapter.py \
  tests/test_phase1_cross_engine_instrument_parity_2x2.py

# Requires a local Python environment containing torch; still Isaac-free.
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_isaac_bank_exam_phase_b.py \
  hope_training/whole_body_tracking/tests/test_physical_ball_helpers.py \
  hope_training/whole_body_tracking/tests/test_face_sign_per_clip.py
```

The Phase-B test file covers bit-exact impulse delegation, blade-disc anti-tunneling, explicit
truth publication, callback-degradation refusal, content/source/target binding, nested scorecard
truth replacement, and the default-off Phase-A lane.

Local verification on 2026-07-11: the focused set completed `69 passed, 1 skipped`; the one skip is
the intentionally artifact-gated simulator acceptance test below. The broader
`whole_body_tracking/tests` run completed `551 passed, 20 skipped, 2 failed`; both failures are an
unrelated existing `MotionLoader` single-`PosixPath` normalization defect in
`test_reward_flags_mdp.py`, outside the Phase-B diff. No Isaac runtime result was produced.

## Historical Phase-B Command — Forbidden

The former clean-detached command for rider SHA `1af7a0b3...b376` is intentionally no longer
copy-pastable here. It is forbidden on training, evaluation, Pod and local simulator checkouts:
the target checkpoint cannot distinguish masked from unmasked 62-D command observations, and a
later metadata backfill cannot repair that missing training-time fact. The current adapter rejects
the rider by content SHA before source/profile validation. Git history preserves the command only
for forensic reconstruction; it is not launch authority.

Recovery requires a post-provenance-epoch checkpoint, a new preregistration, a new Phase-B rider and
all four cells rerun. No field in the frozen files may be edited or refreshed in place.

Historical implementation note only: the revoked checkpoint also predates T1 event-timing fields,
and its rider allowed one narrow neutral-default hydration. That compatibility seam does not
override the content-SHA revocation and may not be exercised to produce a current scorecard. The
old artifact-gated simulator acceptance test must remain unset for current work.

## Validate The Preregistration

This command is local and starts no simulator:

```bash
python3 scripts/validate_phase1_cross_engine_instrument_parity_2x2.py \
  --config configs/phase1_cross_engine_instrument_parity_2x2_prereg_20260711.json
```

The accepted current outcome is a nonzero refusal containing `preregistration is revoked for the
current exact lane`; no evidence or closed-gate JSON may be emitted. A successful validation of the
old paper is an error.

## Runtime Evidence Contract

After a replacement post-epoch paper and matching MuJoCo state export exist, run each cell from one
clean, detached evaluation checkout containing the exact source hashes bound by the **new**
preregistration.
Do not mutate a live training checkout. Normalize each output to
`hope.cross-engine-instrument-cell.v1`, preserving the raw source artifact SHA. Every cell must
contain:

1. the frozen checkpoint, bank, schedule and question-order bindings;
2. a numeric ready state and 100 ordered, uncensored numeric question-state snapshots;
3. exact/fresh lineage;
4. for physical cells, actual physical paddle contact and post-contact flight outcomes;
5. for analytic cells, the frozen capture/contact/flight outcomes.

Create a content-addressed `hope.cross-engine-instrument-parity-evidence.v1` manifest that lists
the four cell files relative to one external artifact root. Validate it with:

```bash
python3 scripts/validate_phase1_cross_engine_instrument_parity_2x2.py \
  --config /path/to/new_post_epoch_preregistration.json \
  --evidence /path/to/instrument_parity_evidence.json \
  --artifact-root /path/to/instrument_parity_cells
```

The validator verifies every file SHA and every row. It emits
`instrument_parity_gate_closed=true` only after accepting exactly four cells. A missing cell,
duplicate cell, virtual-only physical cell, changed question order, mismatched same-engine ready
state, non-finite state, or incomplete physical outcome raises an error and emits no closed gate.

## Runtime Blockers

1. Train/select a post-epoch checkpoint whose contract binds the command-observation mask fact.
2. Freeze a new preregistration and a new Phase-B rider; never reuse the revoked rider or schedule
   identity as current-exact evidence.
3. Export/normalize MuJoCo's signed face-normal vector and complete state schema. The old strike
   ledger contains physical and `cf_*` outcomes but predates this state contract.
4. Produce all four immutable cell artifacts and their evidence manifest.

Until all four are resolved, this operation remains a diagnostic prerequisite and cannot close a
cross-engine model-selection, plant, continuity, deployment, or real-robot gate.
