# Run the air-swing spatial-retarget proposal screen

Date: 2026-07-12; signed-face source rebind 2026-07-13
Scope: CPU-only, no GPU/Pod training and no real robot.  This operation proposes
candidate phases/stations; it does not approve a motion.

## What is frozen

The preregistration is
`configs/motion_video_spatial_retarget_prereg_20260712.json`, SHA-256
`0f757c8c4abfc9bf5070b7db79f494fa1d97a45ddb222609898662eff63af66a`.
The earlier `d8c918ac...5a9f` preregistration was never executed and is revoked: it bound the
historical unsigned-plane scorer that cannot distinguish `n` from `-n`. Do not run or reconstruct
that SHA.
It consumes the exact 792,241-byte v5 result (`c299b7a0...`) and all ten motions.
Franco backhand-loop B/C are ranked first because v5 found intrinsic evidence,
but no motion is omitted.
The proposal ID also binds the source-motion SHA and predecessor-result SHA, so
the same asset name/frame cannot reuse a certificate after its bytes change.
The scorer implementation is pinned at `9d01da15...0f5ec`, and the proposal tool at
`d053dd50...5259b`, together with the
venue physics and explicit `9.5 cm` capture / `0.3 m/s` approach / `10 ms x 100`
rollout constants.

The 2026-07-13 runtime-input fix reads `capture_table_pose_observed=false` from the accepted
result's `frame_contract`; `frame_contract_evidence` is only the content-addressed pointer.  The
previous validator incorrectly required the claim on that pointer and rejected the real accepted
artifact before screening.  Missing/true claims still fail closed.

Returnability also binds the [signed-face contract](../interfaces/racket_contact_geometry.md):
forehand/backhand map physical B to raw A with `[+1,-1]`; achieved and demanded raw-A normals must
share a strict signed hemisphere and both selected physical-B normals must face `+X` before
`orient_normal`. The latter remains impulse-plane math only. A face-blind result is not a proposal.

For one `(motion, immutable question, source frame)` candidate the only allowed
edit is one transform applied atomically to the **whole** trajectory:

- R0: XY translation only;
- R1: the frozen yaw grid `[-10,-5,0,5,10] deg`, then XY translation;
- no z translation, scaling, reflection, joint edit or per-frame transform;
- station bounds: norm `<=0.30 m`, `|x|<=0.20 m`, `|y|<=0.30 m`.

The transform is a counterfactual planner station requirement in the canonical
HOPE table.  It is never a camera/table extrinsic recovered from the room where
the air swing was recorded.

## Validate the tracked contract

```bash
PLAN=configs/motion_video_spatial_retarget_prereg_20260712.json
PLAN_SHA=$(shasum -a 256 "$PLAN" | awk '{print $1}')
CUDA_VISIBLE_DEVICES= python3 scripts/screen_motion_spatial_retarget.py \
  --manifest "$PLAN" --expected-manifest-sha256 "$PLAN_SHA" validate
```

Expected: `PASS (proposal ready, promotion blocked)`.  The 2026-07-12 laptop
verification also passed `pytest -q tests/test_motion_spatial_retarget_screen.py`
(`7 passed`).

## Restore the ignored predecessor without changing it

The laptop checkout intentionally does not contain the full v5 result.  Restore
it under a new ignored path and verify before screening:

```bash
V5=/private/ignored/motion_video_intake_20260711/phase_safety_v5/phase_safety_result.json
test "$(wc -c < "$V5" | tr -d ' ')" = 792241
test "$(shasum -a 256 "$V5" | awk '{print $1}')" = \
  c299b7a04417e855005ad315b40203204bb0cc192398d83980179b212e6bef53
```

Do not overwrite the accepted Pod1 v5 directory, do not edit the Phase-1
training checkout, and do not use basename-only copies assembled from different
motion generations.

## Generate proposals

Choose a new output path; the tool refuses overwrite:

```bash
OUT=/private/ignored/motion_video_intake_20260712/spatial_retarget/proposals_v1.json
CUDA_VISIBLE_DEVICES= python3 scripts/screen_motion_spatial_retarget.py \
  --manifest "$PLAN" --expected-manifest-sha256 "$PLAN_SHA" screen \
  --predecessor-result "$V5" --output "$OUT"
```

The scorer starts flight at the immutable question's ball position.  The
station transform aligns racket/question XY; unchanged z supplies the strict
contact residual.  A result with
`status=complete_proposals_only_promotion_blocked` is the expected current
outcome, not a failure and not an accepted motion.

2026-07-13 的 accepted Pod1 proposal 文件为
`/workspace/codexschema/motion_spatial_retarget_signed_a4bbbaa_v1/proposals.json`，225,920 bytes，
SHA-256 `69c3db16fa78f526aef49f20eeafe0d7e5e3004c4ed27f5e2823bb3574e2465c`。它含 22 个
certificate 前提案：B=19、C=3；accepted/certified 仍均为 0。tracked 摘要是
`configs/motion_video_spatial_retarget_signed_results_20260713.json`。

## Select exactly one B and one C primary

This post-screen selection is CPU-only. It does not rerun the scorer. The
contract binds the exact 225,920-byte proposal file, removes only identical
`yaw=0` aliases between the translation-only tier and the yaw-plus-translation
tier (keeping the translation-only row), and then freezes both the primary and
the remaining fallback order.

```bash
SELECT_PLAN=configs/motion_backhand_loop_bc_proposal_selection_prereg_20260713.json
SELECT_PLAN_SHA=$(shasum -a 256 "$SELECT_PLAN" | awk '{print $1}')
test "$SELECT_PLAN_SHA" = 691fd516477a8d7b56aa9fb562a76684e421f6050e447d707467ee267b0b9b8c

CUDA_VISIBLE_DEVICES= python3 scripts/select_motion_spatial_retarget_candidates.py \
  --prereg "$SELECT_PLAN" \
  --expected-prereg-sha256 "$SELECT_PLAN_SHA" validate

PROPOSALS=/workspace/codexschema/motion_spatial_retarget_signed_a4bbbaa_v1/proposals.json
test "$(wc -c < "$PROPOSALS" | tr -d ' ')" = 225920
test "$(shasum -a 256 "$PROPOSALS" | awk '{print $1}')" = \
  69c3db16fa78f526aef49f20eeafe0d7e5e3004c4ed27f5e2823bb3574e2465c

OUT=/private/ignored/motion_video_intake_20260712/spatial_retarget/bc_selection_v1.json
test ! -e "$OUT"
CUDA_VISIBLE_DEVICES= python3 scripts/select_motion_spatial_retarget_candidates.py \
  --prereg "$SELECT_PLAN" \
  --expected-prereg-sha256 "$SELECT_PLAN_SHA" select \
  --proposals "$PROPOSALS" --output "$OUT"
test "$(shasum -a 256 "$OUT" | awk '{print $1}')" = \
  8a80a409ca69e2fa73757b139b8496bb9cdda2e6a66d3fab48412051b408d2be
```

The exact primaries are B `98e7b883...f3c14` and C
`aa0c86fd...f299`. The result remains certificate-blocked and is byte-identical
to `configs/motion_backhand_loop_bc_proposal_selection_results_20260713.json`.

Do not hand-pick a later row. If and only if a selected candidate later fails
the table/net external-geometry clearance, use `resolve` with outcome code
`external_geometry_table_or_net_clearance_failure`; the command publishes a
new no-clobber decision ledger. A schema-2 materialization, L0 static audit,
vendor-L1 self-collision, or internal dynamics/balance failure stops that asset
and must use its exact stop code from the tracked selection contract. Unknown
codes fail closed. A successful certificate stage continues the same
candidate and does not call `resolve`.

## Promotion remains deliberately blocked

The current manifest says `certificate_bundle_preregistered=false`; passing an
ad-hoc certificate bundle is rejected.  Before any proposal can become an
accepted spatial retarget, create a new content-bound preregistration and prove,
for the exact candidate ID and exact whole-trajectory transform:

1. runtime-order schema-2 materialization;
2. L0 `audit_motion_npz.py` PASS;
3. L1 vendor-MJCF `audit_self_collision.py` PASS;
4. full-trajectory table/net swept clearance with zero hard failures and at
   least `5 mm` minimum clearance.

Dynamics/balance still follows these four gates.  TOPP remains paused; RL and
hardware remain unauthorized.  Final motion/library acceptance is the AgiBot
vendor MuJoCo Gate3/Gate3B no-reset test.
