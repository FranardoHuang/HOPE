# Run The Phase-1 Cross-Engine Instrument-Parity 2x2

## Current Status

`Preregistered / Phase-B source complete / runtime unvalidated`. Do not count or launch this
paper from a training checkout. The prerequisite contract is
`configs/phase1_cross_engine_instrument_parity_2x2_prereg_20260711.json`; its validator must
continue to report `instrument_parity_gate_closed=false` until all four real cells exist.
The Isaac rider is separately frozen by
`configs/phase1_isaac_bank_exam_physical_truth_phase_b_contract_20260711.json`. Source completion
does not constitute simulator evidence: its `runtime_claim_now` remains false until a clean,
detached evaluation checkout produces a complete scorecard.
The current preregistration SHA is `bd90f6f2...0175`, validator SHA is
`eb1b2fa6...f4e4`, and Phase-B rider-contract SHA is `1af7a0b3...b376`.

This operation is simulator-only. It authorizes no real-robot command or deployment test and does
not change, stop, promote, or restart a training arm.

## 这张卷为什么存在

同一组 100 道题曾在两引擎里得到不同排名，但双方已有的回球列实际上都是解析判分，且引擎执行和
判分仪器没有被干净拆开。尤其是当前 Python MuJoCo BankExam 没有仿真中的球拍—球接触：它只物理
推进机器人，再把击球时的球拍状态交给 `VirtualReturnScorer`。因此“同一题序”不是物理一致性
证据。冻结的 2×2 要补齐四个真实格子，分别解释引擎执行差异和判分仪器差异：

| 引擎 | 物理真值 | 解析对照 |
| --- | --- | --- |
| Isaac | 必须补；Phase-B 只有源码，运行未验证 | 已有解析诊断仍需标准化、内容绑定的格子输出 |
| MuJoCo | 必须补；当前 BankExam 没有物理格 | 已有 Python BankExam/`cf_*` 解析诊断仍需标准化、内容绑定的格子输出 |

任一行或列都不能替代另一格。尤其是把 Isaac 解析回球改名成“物理回球”时，校验必须直接失败。

## Frozen Paper

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

## Clean-Detached Isaac Phase-B Command

This command is **not** authorized on either live training checkout. Run it later only from a
clean detached evaluation worktree whose files match every SHA in the Phase-B contract; keep all
outputs outside that worktree. The frozen `PHASE_B_CONTRACT_SHA` is shown explicitly below.

```bash
PHASE_B_CONTRACT_SHA=1af7a0b3589d57bfbd2da0b8af6130641298b647e4d80e52b5ef673a84e5b376
hope_isaac_py hope_training/whole_body_tracking/scripts/isaac_bank_exam.py \
  task=HOPEPingPongVirtualBall headless=true device=cuda:0 \
  +run_dir=/workspace/codexschema/phase1_fresh_20260711/runs/phase1_fresh_v3_S1_seed1 \
  checkpoint=/workspace/codexschema/phase1_fresh_20260711/runs/phase1_fresh_v3_S1_seed1/model_2000.pt \
  +exam_bank=/ABSOLUTE/PATH/TO/s1_v4rg_v3_exam.npz \
  +schedule_json=/workspace/codexschema/phase1_fresh_20260711/q50/fresh_SZ_seed1_model2000_vs_model4000_exact_v1/shared_clean_k100.schedule.json \
  +per_clip_quota=50 +schedule_seed=0 +noise_scale=0.0 \
  +instrument_physical_truth_phase_b=true \
  +phase_b_contract=/ABSOLUTE/DETACHED/REPO/configs/phase1_isaac_bank_exam_physical_truth_phase_b_contract_20260711.json \
  +expected_phase_b_contract_sha256=${PHASE_B_CONTRACT_SHA} \
  +output_dir=/workspace/codexschema/phase1_fresh_20260711/instrument_parity/isaac_phase_b
```

The evaluator audits clean/detached state and all bound source hashes both before launch and after
the last attempt. It refuses `allow_inexact_contract=true` in this cell. A successful process is
still only the Isaac-physical quarter of the 2x2, not a closed parity gate.

The checkpoint predates the T1 event-timing fields now present in the evaluator class. The bound
profile permits only their neutral default materialization (`event_timing_mode=disabled`, empty
schedule/SHA, no repeat); generic exact hydration remains strict for every other field. Any
non-neutral T1 value invalidates this paper.

The marked simulator-dependent acceptance check is intentionally skipped until that scorecard
exists:

```bash
HOPE_PHASE_B_ISAAC_SCORECARD=/path/to/isaac_bank_exam.json \
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_isaac_bank_exam_phase_b.py \
  -k simulator_dependent
```

## Validate The Preregistration

This command is local and starts no simulator:

```bash
python3 scripts/validate_phase1_cross_engine_instrument_parity_2x2.py \
  --config configs/phase1_cross_engine_instrument_parity_2x2_prereg_20260711.json
```

The accepted pre-runtime output is `valid_preregistered_runtime_blocked` and
`instrument_parity_gate_closed=false`. Any other current claim is an error.

## Runtime Evidence Contract

After Isaac Phase B and the matching MuJoCo state export exist, run each cell from one clean,
detached evaluation checkout containing the exact source hashes bound by the preregistration.
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
  --config configs/phase1_cross_engine_instrument_parity_2x2_prereg_20260711.json \
  --evidence /path/to/instrument_parity_evidence.json \
  --artifact-root /path/to/instrument_parity_cells
```

The validator verifies every file SHA and every row. It emits
`instrument_parity_gate_closed=true` only after accepting exactly four cells. A missing cell,
duplicate cell, virtual-only physical cell, changed question order, mismatched same-engine ready
state, non-finite state, or incomplete physical outcome raises an error and emits no closed gate.

## Runtime Blockers

1. Run the locally completed Phase-B rider from a clean detached evaluation checkout on the frozen
   q50 schedule; no accepted runtime scorecard contains it yet.
2. Run the new Isaac numeric instrumentation on the frozen q50 schedule; no accepted runtime
   scorecard contains it yet.
3. Export/normalize MuJoCo's signed face-normal vector and complete state schema. The old strike
   ledger contains physical and `cf_*` outcomes but predates this state contract.
4. Produce all four immutable cell artifacts and their evidence manifest.

Until all four are resolved, this operation remains a diagnostic prerequisite and cannot close a
cross-engine model-selection, plant, continuity, deployment, or real-robot gate.
