# Archive Phase-1 q10 checkpoint curves

Use `scripts/collect_phase1_q10_curve_results.py` after copying completed curve-worker
evidence to a local, read-only evidence directory. The collector is a post-hoc validator:
it does not SSH to a Pod, inspect a live process, open a checkpoint with Torch, run a judge,
signal a process, alter a training/evaluation checkout, or authorize a real-robot command.

The output is only a checkpoint-growth direction screen. It cannot stop or promote a run,
select a deploy checkpoint, trigger q50, or claim a formal gate. Any such decision needs the
separate preregistered immutable q50 paper.

## Required copied evidence

For every selected manifest job, copy these four immutable files without changing the Pod:

1. the hardened curve-worker state JSON;
2. the generated `judge_report_model_*.md`;
3. that report's `exam/mujoco_sim2sim_summary.json` scorecard; and
4. an explicit read-only checkpoint-audit sidecar described below.

Also retain the exact worker manifest bytes. Record every copied file's SHA-256 in a local
archive index. Paths in the index may be absolute or relative to the index file. The collector
reads only those explicit paths; it never searches a run directory and never chooses a
"latest" report.

The worker state must contain the hardened bindings written by
`phase1_checkpoint_curve_worker.py`: `manifest_sha256`, `job_spec_sha256`,
`job_contract_sha256`, `checkpoint_sha256`, `judge_script_sha256`, `training_commit`, and
`eval_commit`. A pre-hardening/legacy state is evidence, but cannot be laundered into an
archive by this tool.

## Explicit checkpoint-audit sidecar

The sidecar supplies facts that a JSON/report collector cannot safely infer without opening
the checkpoint. Its root is exactly:

```json
{
  "schema_version": 1,
  "audit_kind": "phase1_q10_checkpoint_audit",
  "read_only": true,
  "real_robot_commands": false,
  "job_id": "...",
  "manifest_sha256": "...",
  "state_sha256": "...",
  "job_spec_sha256": "...",
  "job_contract_sha256": "...",
  "judge_report_sha256": "...",
  "scorecard_sha256": "...",
  "checkpoint": {
    "path": "/absolute/original/run/model_19000.pt",
    "sha256": "...",
    "filename_iteration": 19000,
    "embedded_iteration": 19000,
    "floating_tensor_count": 74,
    "nonfinite_floating_elements": 0,
    "all_floating_tensors_finite": true,
    "embedded_training_contract_sha256": "...",
    "embedded_training_contract_lineage_exact": false
  },
  "training_contract": {
    "path": "/absolute/original/run/params/training_contract.json",
    "sha256": "...",
    "schema_version": 3,
    "structure_validator": "pass",
    "binding_validator": "pass",
    "lineage_exact": false
  },
  "provenance": {
    "training_commit": "...",
    "eval_commit": "...",
    "judge_script_sha256": "...",
    "evaluator_source_sha256": "..."
  },
  "evaluation": {
    "immutable_schedule_sha256": "...",
    "evaluation_contract_exact": false
  }
}
```

Generate this sidecar only from a read-only checkpoint audit. The collector verifies its
hash and all repeated edges, but deliberately does not pretend that a user-authored boolean
is a substitute for the checkpoint audit itself.

## Archive index

The index is strict schema-version 1 JSON with no extra root keys:

```json
{
  "schema_version": 1,
  "archive_id": "phase1_M2_seed2_19000_q10",
  "screen_policy": {
    "screen_only": true,
    "stop_or_promote_allowed": false,
    "q50_triggered": false,
    "decision_claim": null
  },
  "manifests": [
    {
      "id": "causal_pod2",
      "path": "phase1_checkpoint_curve_scaleout_causal_pod2_20260711.json",
      "sha256": "...",
      "barrier_ids": ["causal_19000"]
    }
  ],
  "pairs": [
    {
      "id": "M2_seed2_face_pair_19000",
      "kind": "face_pair",
      "members": [
        "causal_pod2:M2_old_seed2_19000_clean_q10",
        "causal_pod2:M2_S1_seed2_19000_clean_q10"
      ]
    }
  ],
  "evidence": [
    {
      "manifest_id": "causal_pod2",
      "job_id": "M2_old_seed2_19000_clean_q10",
      "worker_state": {"path": "old.state.json", "sha256": "..."},
      "judge_report": {"path": "old.report.md", "sha256": "..."},
      "scorecard": {"path": "old.summary.json", "sha256": "..."},
      "checkpoint_audit": {"path": "old.audit.json", "sha256": "..."}
    }
  ]
}
```

The example abbreviates the second evidence row; a real index must include it. Evidence must
exactly cover every job in every selected barrier. Pairs must partition those jobs exactly
once and contain two members. Supported pair kinds are:

- `face_pair`: same family/seed/plant/milestone, legacy versus shared face pairing;
- `plant_pair`: same family/seed/face/milestone, zero versus nonzero plant flag; and
- `seed_replication_pair`: same family/cell/face/plant/milestone, two distinct training seeds.

For the scale-out causal 19k archive, select `causal_19000` from one family manifest and use
its old/shared `face_pair`. M2 and M3 use different exam banks/schedules, so archive them
separately.

For the scale-out fresh 2k archive, one index may bind both Pod manifests at `fresh_2000` and
cover all fourteen selected jobs with seven explicit `seed_replication_pair` entries. The
collector will accept this only if all scorecards carry the identical exam bank, immutable
schedule SHA, and exact ordered question identity. If the copied evidence does not form that
complete same-paper partition, archive smaller complete same-paper pairs rather than filling
or guessing missing rows.

## Validate and write

Validation is dependency-light and performs no write:

```bash
python3 scripts/collect_phase1_q10_curve_results.py \
  --index /local/evidence/archive_index.json \
  --validate-only
```

After validation, write the content-addressed aggregate under a local archive root:

```bash
python3 scripts/collect_phase1_q10_curve_results.py \
  --index /local/evidence/archive_index.json \
  --output-dir /local/phase1_q10_archives
```

The filename is `<archive_id>_<content_sha256>.json`. `content_sha256` is the SHA-256 of the
canonical JSON `content` object stored inside the document. Re-running with the same evidence
is idempotent; pre-existing different bytes at the same content address are refused.

## Fail-closed checks

The collector refuses, among other cases:

- any schedule other than clean K=20, ten attempts per side, seed 0, noise 0;
- q50 or an index that claims stop, promotion, q50 triggering, or another decision;
- a missing/extra job, partial barrier, incomplete/overlapping pair, or mixed schedule/bank;
- a worker state that is not complete `returncode=0` or lacks hardened SHA bindings;
- checkpoint filename/embedded iteration disagreement, non-finite tensors, or a broken
  checkpoint-to-adjacent-schema-3 hard-contract binding;
- report, scorecard, state, sidecar, manifest, job, job-contract, judge, training commit,
  eval commit, evaluator source, exactness, or schedule SHA disagreement; and
- censored/ineligible questions or per-side/aggregate metrics that are not computed from all
  10/10/20 scheduled attempts.

The dependency-light synthetic fixture is under
`tests/fixtures/phase1_q10_curve_collector/`; verify it with:

```bash
python -m pytest -q tests/test_collect_phase1_q10_curve_results.py
```
