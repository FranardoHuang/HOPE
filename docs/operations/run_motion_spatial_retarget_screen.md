# Run the air-swing spatial-retarget proposal screen

Date: 2026-07-12; signed-face source rebind 2026-07-13
Scope: CPU-only, no GPU/Pod training and no real robot.  This operation proposes
candidate phases/stations; it does not approve a motion.

## What is frozen

The preregistration is
`configs/motion_video_spatial_retarget_prereg_20260712.json`, SHA-256
`69bdeabc9b5a934143c52ec6a7fe28ab0a0be6573b2f14f0748e49063c69eb62`.
The earlier `d8c918ac...5a9f` preregistration was never executed and is revoked: it bound the
historical unsigned-plane scorer that cannot distinguish `n` from `-n`. Do not run or reconstruct
that SHA.
It consumes the exact 792,241-byte v5 result (`c299b7a0...`) and all ten motions.
Franco backhand-loop B/C are ranked first because v5 found intrinsic evidence,
but no motion is omitted.
The proposal ID also binds the source-motion SHA and predecessor-result SHA, so
the same asset name/frame cannot reuse a certificate after its bytes change.
The scorer implementation is pinned at `9d01da15...0f5ec`, and the proposal tool at
`43eccb43...289b4`, together with the
venue physics and explicit `9.5 cm` capture / `0.3 m/s` approach / `10 ms x 100`
rollout constants.

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
