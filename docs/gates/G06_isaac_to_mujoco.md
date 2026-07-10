# G06 Isaac-To-MuJoCo Parity

Status: Partial (parity procedure operational and used to gate the 2026-07-02 sim-to-real; formal per-checkpoint acceptance thresholds still to be recorded)

## Goal

Test whether a policy learned in Isaac can be replayed or approximated in MuJoCo.

This gate is the sim-to-sim bridge before real deployment.

## Inputs

- Isaac-trained policy ONNX from G05 (exported with the full metadata contract).
- MuJoCo A3 model from G04 (`a3_pingpong.xml`).
- Shared joint order and observation/action contract (`docs/interfaces/policy_observation_action.md`).

## Outputs

- Replay/evaluation procedure with Isaac-exact metrics.
- Cross-sim metrics and known mismatch list.
- Decision on which MuJoCo configuration is deploy-faithful.

## Related Directories

- `hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py` — the parity evaluator.
- `agi/a3_deploy_example/` — active deploy tree: `MUJOCO_VALIDATION_RUNBOOK.md`, `SIM_DEPLOY_REHEARSAL.md`, `SIM_FIDELITY_NOTE_FOR_AGI.md`.
- `agi/A3_MuJoCo_Sim/` — vendor AimRT MuJoCo sim (the explicit-PD subscriber lives here).
- `agi/code_deployment/a3_deploy_example/` — older vendor reference subset.

## Operation Docs

- [../operations/run_training.md](../operations/run_training.md)
- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)
- [../operations/run_shared_interface_rehearsal.md](../operations/run_shared_interface_rehearsal.md)

## Acceptance Criteria

- The same action ordering is verified in both simulators.
- The exported deploy ONNX (not a re-export) runs in MuJoCo with the training observation rebuilt exactly.
- Divergence sources are documented: contact, latency, actuator, timestep, observation delay, model mismatch.
- Exact-strike metrics from Isaac are reproduced in MuJoCo and recorded per accepted checkpoint.

## Current State

Done (2026-06-27 → 2026-07-02, recorded 2026-07-03):

- The parity procedure exists and is battle-tested: `scripts/mujoco_eval_onnx.py` loads the exact
  exported deploy ONNX, reads the whole actuator contract from ONNX metadata (joint_names,
  default_joint_pos, action_scale, kp/kd, body_names — fails loudly if missing), auto-detects the
  175-D deploy-parity vs 180-D legacy obs contract, rebuilds the Isaac actor observation in MuJoCo
  (same frame math; the deploy-honest racket-target reframe is verified by
  `scripts/realsensor_obs_reference.py`), and reproduces Isaac's exact-strike metrics
  (pos/vel/normal pass, composite, hit-speed error, velocity attainment) with per-clip
  forehand/backhand breakdowns and per-step CSVs.
- The dominant divergence source was isolated to actuator PD integration: with the same ONNX and
  byte-identical `a3_pingpong.xml`, MuJoCo with `implicitfast` + kd in `dof_damping` (Isaac
  `ImplicitActuator` equivalent) is stable with clean swings, while the AGI deploy sim's
  explicit-Euler PD path (`joint_actuator_subscriber.cc`, MJCF without an integrator attribute,
  passive damping not zeroed) diverges within ~0.1 s. Switching only the PD integration moved
  hit-speed error 0.61 → 0.31 m/s and velocity attainment 0.35 → 0.88. One-flag reproduction:
  `--pd-mode implicit` vs `--pd-mode explicit --keep-passive`. See
  `agi/a3_deploy_example/SIM_FIDELITY_NOTE_FOR_AGI.md`.
- Current verdict stance (2026-07-02): implicit PD remains the Isaac-faithful cross-check, but the
  binding pre-hardware gate is the AGI explicit clipped-PD MuJoCo run ("falls in MuJoCo = falls on
  the real robot"). The deployed policy was fine-tuned to survive it
  (`launch_explicitpd_ft.sh`, exported via `export_onnx_explicitpd.sh`).
- A deploy-faithful episode protocol exists: `--deploy-faithful` mirrors the C++ runner
  (nominal-stand start, windup hold with pinned time_to_strike, one full clip per swing, rest
  between swings, no teleports, absolute fall terminations only), reporting swing completion rates
  and time-to-fall.
- Eval mode B exists (2026-07-04): `--target-source venue-balls` (`mujoco_eval_onnx.py` +
  `scripts/venue_ball_sampler.py`) samples fitted venue incoming balls (with spin), StrikeSpec-
  inverts the demanded racket state (pos/vel/normal, sign-matched to the swing side's reference
  face), drives the unchanged target pipeline, and scores a virtual return at the exact-strike
  frame (capture gate → venue contact model → drag+Magnus flight → bounds + net clearance).
  Headline reported as `return_success_rate` per strike; mode-A (`boxes`) output stays
  byte-identical. First run: pos/vel tracking survives the OOD venue distribution (3.7 cm /
  0.18 m/s) but the face normal is clip-locked (36-76° err, 0% legal returns) — the 175-D
  contract has no normal channel (`docs/motion_and_contract_v3.md`). v1 caveats: uncorrelated
  box sampling, human-receiver contact heights (0.98-1.26 m vs trained 0.72-1.13 m —
  intentional realism, expect pos_pass to drop), incompatible with `--deploy-faithful`.
- The normal counterfactual is a committed output (2026-07-05; was an ad-hoc uncommitted
  analysis on 07-04): every venue strike is auto-rescored with the DEMANDED face normal swapped
  into the achieved kinematics — `cf_*` columns after the 14 venue columns + a CF summary
  block. Committed record (P2 product line, 9600 steps seed 0, 44 strikes): actual 0/44 vs
  counterfactual 44/44, CF median landing error 0.10 m; the 07-04 2400-step run reproduces
  byte-identically (first 43 CSV columns). The face-orientation channel alone fails the return.
- Fixed-normal inversion exists and delivered a verdict (2026-07-05): `--venue-fixed-normal`
  pins the StrikeSpec normal at the clip reference face (`solve_fixed_normal`, velocity-only
  LM; free `solve()` untouched; 16/16 planner tests). Result: the path-A ceiling is ~0% — a
  brute-force reachability scan (face pinned, all |v_r| ≤ 6 m/s, ~7k landings/ball) shows the
  forehand face ([0.41,0.90,-0.17], near-sideways) lands x ≤ 1.4 m at ANY racket velocity
  (never clears the net at 1.87 m) and the backhand face only reaches a net-hugging cross-court
  sliver (x≈1.9-2.0, |y|≈0.3-0.67) outside the legal landing box (≥0.3 m depth guard =
  training's own dink rule). Premise verified: mode-A achieved normal is within 1.9° of the
  clip reference, so the pinned face IS the policy's face. Planner adaptation cannot rescue the
  clip-locked face; the normal-channel contract change (175→179) is the only path.
  Evidence: pod `/workspace/franco/cf_eval/` (scan_reachability.py, modeB_*.log).
- A deploy-parity mid-swing switch stress protocol exists (2026-07-05): `--switch-stress P`
  (multiswing only; default off = byte-identical) aborts the swing each step with probability P
  exactly like the deploy runner's planner re-decides (training `clip_switch` semantics:
  uniform new clip, windup frame, fresh hold + target, robot untouched; tracking guards off —
  balance falls + timeout only). Reports switches, falls, 2 s post-switch survival, post-switch
  vs clean-swing hit rates. First matrix ({P2, R11} × {implicit, explicit+keep-passive} ×
  {~0, 0.002, 0.01}/step, 24000 steps each): zero falls in all 12 runs, 100% post-switch
  survival, post-switch hit rate ≈ clean — the switch discontinuity alone does not topple even
  the non-switch-trained P2 in MuJoCo; R11's in-distribution hit-rate tax remains visible on
  the explicit gate (0.98-0.99 vs P2's 0.99-1.00). Logs: pod `/workspace/franco/cf_eval/sw_*`.
- A documented validation flow with an acceptance-criteria table exists:
  `agi/a3_deploy_example/MUJOCO_VALIDATION_RUNBOOK.md` (rate ~50 Hz, sync stable, infer < 20 ms,
  projected gravity sanity, bounded actions, neck passive).

Not done:

- Formal per-checkpoint acceptance: the metric thresholds and the numbers for the currently shipped
  checkpoint (`model_p4_deployparity` / explicitpd_ft `model_25700`) are not yet pasted into this
  gate as an accepted record.
- (Fixed 2026-07-03, branch `audit-leftover-fixes`.) `eval_realsensor_hopex.sh` /
  `export_onnx_explicitpd.sh` now resolve their own location and take `HOPE_EVAL_*` /
  `HOPE_EXPORT_*` env overrides, and `mujoco_eval_onnx.py` resolves strike phases as CLI >
  ONNX `clip_strike_phases` metadata > built-in legacy `(0.36, 0.50)` (plus a
  `clip_seg_lengths`-vs-npz mismatch warning). The `--onnx`/`--motion-files` defaults still point
  at a legacy run — pass current artifacts explicitly.
- No decision recorded on MuJoCo as a training backend (currently it is a validation/dry-run stage
  only).

## Risks

- A policy can appear valid in Isaac but fail in MuJoCo because of actuator/contact mismatch — this
  happened (explicit-PD divergence) and cost significant time before the root cause was isolated.
- Evaluating with the script's stale defaults silently tests the wrong contract; always pass the
  checkpoint's own clips/phases.

## Next Steps

1. Record the accepted sim2sim numbers for the shipped checkpoint (implicit cross-check + explicit
   clipped-PD gate + `--deploy-faithful` protocol) in this gate.
2. When the mocap→planner bridge lands, extend the MuJoCo rehearsal to consume live
   `/racket/command` targets instead of sampled planner-equivalents
   (`docs/operations/run_shared_interface_rehearsal.md`).

## Audit update 2026-07-10: formal BankExam ruler

The old headline scores are not a trustworthy promotion ruler. The evaluator
had an exact-strike one-step offset, omitted pre-strike failures from its
denominator, compared different question slices across noise columns and did
not enforce the held-out split. These are now closed:

- one immutable schedule with stable question IDs and per-attempt seeds;
- all scheduled attempts remain in the denominator;
- every noise/model column receives the same ordered questions;
- train/exam split, motion SHA/order/frame and physics-source lineage are
  fail-closed;
- every formal attempt starts from the MJCF named `stand` keyframe with all
  hidden state and last action reset; teacher-reference reset is diagnostic;
- schedule, ready-state, MJCF and resolved execution-contract SHA are emitted
  in summaries and attempt CSVs;
- actuator integration, armature, ctrl/velocity limits and q-des contract come
  from schema-v3 rather than observation width guesses.

Non-zero PhysX joint friction has no exact MuJoCo `frictionloss` equivalent.
Formal BankExam therefore refuses it. `--allow-inexact-contract` may run a
direct-number proxy, but the result is stamped
`evaluation_contract_exact=false` and cannot be booked. Here `exact` means the
listed execution protocol is bound; it does not claim complete cross-engine
dynamics equivalence.

All key historical scores must be rerun after fresh export; retain old values
only with an explicit `old scorer` label.

The 2026-07-11 local Phase-1 snapshot also contained a NumPy
`virtual_return_scorer.py` and a saved-run `termination_contract.py`.  They
were retained as simulator-independent specifications only.  They are not
wired into `mujoco_eval_onnx.py` or the venue sampler, so they do not alter the
formal ruler described above.  Production parity remains a pending
schema-v3-adapter gate, with explicit skipped integration assertions rather
than an implicit fallback.

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_bank_exam_schedule.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py
```
