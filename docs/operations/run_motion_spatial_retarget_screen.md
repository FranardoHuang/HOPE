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

## Materialize the two frozen primaries without choosing a fallback

This CPU-only step turns each selected whole-motion transform into a new GMR
pickle. [`SE(2)`](../DEFINITIONS.md) means one proper yaw plus XY translation
applied to every floating-root frame; it is not per-frame editing or TOPP.
The B/C plans are independent and no-clobber. `static` reads repository files,
`inspect` reads the exact private source without writing, and `consume` creates
the motion first and publishes `materialization_report.json` last.

```bash
TOOL=scripts/materialize_motion_spatial_se2.py
test "$(shasum -a 256 "$TOOL" | awk '{print $1}')" = \
  21ebbe68d5d76acde90bb413f68928df9d87cb053275d9f0f46d36dbf1187375

B_PLAN=configs/motion_backhand_loop_b_se2_materialization_prereg_20260714.json
C_PLAN=configs/motion_backhand_loop_c_se2_materialization_prereg_20260714.json
B_SHA=e016ca742dfebbd9726b03df1ad3cd7e75f19a07557e5e458e57e00088751aee
C_SHA=27f938cd6016fcadada8c6ea806329279c379ccb77b5db99b8902275ebd9d454

for row in "$B_PLAN:$B_SHA" "$C_PLAN:$C_SHA"; do
  plan=${row%%:*}
  sha=${row#*:}
  test "$(shasum -a 256 "$plan" | awk '{print $1}')" = "$sha"
  CUDA_VISIBLE_DEVICES= python3 "$TOOL" --prereg "$plan" \
    --expected-prereg-sha256 "$sha" static
  CUDA_VISIBLE_DEVICES= python3 "$TOOL" --prereg "$plan" \
    --expected-prereg-sha256 "$sha" inspect
done
```

Only after both no-write inspections pass may the operator run the same command
with `consume`, one plan at a time. Before each consume, verify that its exact
`output_contract.output_root` does not exist. A pre-existing root is not a
resume point. The report's presence is the completion marker; absence means the
output is incomplete and must not be consumed downstream.

Do not change `candidate_id` or call selection `resolve` here. A materialization
or internal verification failure stops that asset. Only a later independently
recorded table/net external-geometry failure may advance the frozen fallback
ladder. Passing this step still authorizes only a separate schema-2
preregistration; L0, vendor L1, table/net clearance, dynamics, simulator,
training, TOPP and hardware remain blocked.

### Accepted 2026-07-14 runtime receipt

Pod1 CPU-only `consume` has already completed for both plans; do not rerun into either no-clobber output root.
B published a 27,927-byte motion SHA `278279125528c827e0a980389b040d54d16140620c59c67c878286be9d1c8ad6`
and 4,051-byte report SHA `a238c077524586b2f1181cd24cb84ee29aa985ab274cfb43292f3159c0daadf3`.
C published a 30,055-byte motion SHA `0dd981a6d29c0c5321c905d1591a59fbb79763de6e43d92d4d76aefdc29ff48b`
and 4,068-byte report SHA `b3b93d2cdb0a288f04aed764e5fdca92182cee625715a953600439088f59ff67`.
The tracked result ledger is
`configs/motion_backhand_loop_bc_se2_materialization_results_20260714.json`. Restore private payloads by the
exact paths/SHA in that ledger; never regenerate them merely because a local copy is absent.

## B/C schema-2 joint-order source gate (2026-07-14)

Before writing or running the separate B/C schema-2 preregistration, validate the tracked
[GMR](../DEFINITIONS.md) `dof_pos` → runtime articulation `joint_pos` mapping:

```bash
python3 hope_training/whole_body_tracking/scripts/a3_joint_order_contract.py
python3 -m pytest -q tests/test_a3_joint_order_contract.py
```

Expected JSON contains `bijection_valid=true`, `source_equals_target=false`, and
`schema2_materialization_authorized=false`; focused tests currently report `12 passed`. The source
and target tables are respectively `configs/a3_gmr_dof_pos_joint_order.txt` and
`configs/a3_runtime_articulation_joint_order.txt`. Their byte/name SHAs and both permutations are
bound in `configs/a3_joint_order_bijection_v1.json`.
The current contract/validator/converter SHA-256 values are respectively
`b09987ff...4815`, `8f01d20d...1ae9`, and `a151a691...7f04`; any prereg must bind full hashes rather
than these display abbreviations.

For a future exact donor metadata receipt, serialize the ONNX custom metadata map as JSON and add:

```bash
python3 hope_training/whole_body_tracking/scripts/a3_joint_order_contract.py \
  --metadata-json /absolute/path/to/content_bound_onnx_metadata.json
```

All three metadata rows are mandatory: `joint_names`, `articulation_joint_names`, and
`action_joint_ids`; partial legacy metadata must fail. `csv_to_npz_mujoco.py` now consumes the same
contract through `--joint-order-contract` and no longer carries its own GMR list. Do not run that
converter on B/C until a separate no-clobber prereg binds the exact SE(2) PKL, donor metadata,
vendor MJCF/include/mesh closure, runtime body order and output path. Because B/C root trajectories
are already in the HOPE frame, that prereg must select `--hope_frame off`; a second automatic
rotation is forbidden. Passing this source gate does not run forward kinematics, create schema-2,
or authorize L0/L1/simulator/training/hardware.

## Validate the independent B/C schema-2/FK source gate (2026-07-14)

The separate preregistration now exists. [`FK`](../DEFINITIONS.md) means forward kinematics: here it
will later evaluate the frozen joint/root trajectory in the vendor MuJoCo model, without stepping a
dynamics simulation. The current command is `static` only: it reads tracked repository sources,
derives the full MJCF include/external-file closure, and must not read either private pickle or an
ONNX file.

```bash
TOOL=scripts/materialize_motion_schema2_fk.py
TOOL_SHA=33cf23eecff514a0e89bfe245db5b63470c4cd1dc9a433d0b920dfd84b9caebd
B_PLAN=configs/motion_backhand_loop_b_schema2_fk_prereg_20260714.json
B_SHA=3d71cc02c6ae68d0ecedf280e8341d763ad39ec0aac1757367c9719e761d33ae
C_PLAN=configs/motion_backhand_loop_c_schema2_fk_prereg_20260714.json
C_SHA=662b8c4c0851d2f6d9d5c23313dc0c27334528a2b5fb2b62ad90bc3447257e31

test "$(shasum -a 256 "$TOOL" | awk '{print $1}')" = "$TOOL_SHA"
test "$(shasum -a 256 "$B_PLAN" | awk '{print $1}')" = "$B_SHA"
test "$(shasum -a 256 "$C_PLAN" | awk '{print $1}')" = "$C_SHA"

CUDA_VISIBLE_DEVICES= python3 "$TOOL" \
  --prereg "$B_PLAN" --expected-prereg-sha256 "$B_SHA" \
  --peer-prereg "$C_PLAN" --expected-peer-prereg-sha256 "$C_SHA" \
  --hope_frame off static

CUDA_VISIBLE_DEVICES= python3 "$TOOL" \
  --prereg "$C_PLAN" --expected-prereg-sha256 "$C_SHA" \
  --peer-prereg "$B_PLAN" --expected-peer-prereg-sha256 "$B_SHA" \
  --hope_frame off static

python3 -m pytest -q tests/test_motion_backhand_loop_bc_schema2_fk_prereg.py
```

Expected: two `PASS static ... pair_exact=true ... runtime_inspection=false` lines and
`17 passed`. `--hope_frame off` means “the accepted SE(2) root is already in the HOPE world
frame”; the parser has no `on` choice. B/C output roots are disjoint and must not exist.

The shared runtime contract SHA is
`3d32b146e72029960ebf9cb2777f484804dafc87097e9cd3d0513dc277eed6e8`. It binds one vendor XML,
zero includes and 74 referenced meshes under closure SHA `e0381752...962de`. It also binds
`configs/a3_schema2_fk_donor_metadata_v1.json`, but that file is an expected three-row metadata
subset tied to exact donor ONNX SHA `0c428ddf...b7b155`; it is deliberately **not** a claim that the
rows were re-extracted from the ONNX in this source gate.

### Accepted read-only runtime inspection and review-gated consume activation

The exact B/C private files and formal donor were restored without copying over either source or
output root. The successful historical commands used the existing CPU runtime below, one asset at a
time. `inspect` hashes and restricted-loads the pickle, hashes the ONNX and re-extracts its required
metadata, loads the content-bound vendor MJCF, verifies all joint/body names, and writes nothing.
These commands are shown for audit; do not rerun them merely because the receipt is now tracked.

```bash
SRC=/workspace/codexschema/nohope_schema2_fk_inspect_748b6d5
PY=/workspace/hope_mjeval_venv/bin/python
TOOL=$SRC/scripts/materialize_motion_schema2_fk.py
B_PLAN=$SRC/configs/motion_backhand_loop_b_schema2_fk_prereg_20260714.json
C_PLAN=$SRC/configs/motion_backhand_loop_c_schema2_fk_prereg_20260714.json
B_SHA=3d71cc02c6ae68d0ecedf280e8341d763ad39ec0aac1757367c9719e761d33ae
C_SHA=662b8c4c0851d2f6d9d5c23313dc0c27334528a2b5fb2b62ad90bc3447257e31
DONOR=/workspace/codexschema/gate3_face179_b5762fa/isolated_assets/formal_sz_seed3_model2000_11f3a288/exported_2fa3534/policy.onnx
test "$(shasum -a 256 "$DONOR" | awk '{print $1}')" = \
  0c428ddf9968b047acbe7bbd5a39069a8e661ab0421038ea3b635284deb7b155

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PY" "$TOOL" \
  --prereg "$B_PLAN" --expected-prereg-sha256 "$B_SHA" \
  --peer-prereg "$C_PLAN" --expected-peer-prereg-sha256 "$C_SHA" \
  --hope_frame off --donor "$DONOR" inspect

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PY" "$TOOL" \
  --prereg "$C_PLAN" --expected-prereg-sha256 "$C_SHA" \
  --peer-prereg "$B_PLAN" --expected-peer-prereg-sha256 "$B_SHA" \
  --hope_frame off --donor "$DONOR" inspect
```

The default `/usr/bin/python3` first failed closed with rc=2 because `onnxruntime` was missing; it
wrote nothing and is not an accepted inspection. The exact passing runtime was Python/NumPy/
ONNX Runtime/MuJoCo `3.12.3/2.5.0/1.27.0/3.10.0`. B/C then returned rc=0 with `frames=91/98`,
`donor_exact=true`, and `no_write=true`. Source checkout `748b6d5fe24bfe58915c34d8dfe09f254f8e4957`
was detached and clean before/after; both output roots were absent before/after.

Validate the tracked runtime receipt and next-attempt activation from the reviewed source checkout:

```bash
RECEIPT=configs/motion_backhand_loop_bc_schema2_fk_runtime_inspection_receipt_20260714.json
RECEIPT_SHA=8e2d2d2d7a4fe0779104456d3bcb32f03cfda82e831958216eefb0fb35b3fb61
ACTIVATION=configs/motion_backhand_loop_bc_schema2_fk_consume_activation_20260714.json
ACTIVATION_SHA=366d59d51d40111205aa8c8b43e7722218d522b8b568d25772eab1f46f2d6337
VALIDATOR=scripts/validate_motion_schema2_fk_consume_activation.py
VALIDATOR_SHA=3c666f225d389a67a8ef9523004cce0aa6d76bd119cd5f249fa54e14a1c77d72

test "$(shasum -a 256 "$RECEIPT" | awk '{print $1}')" = "$RECEIPT_SHA"
test "$(shasum -a 256 "$ACTIVATION" | awk '{print $1}')" = "$ACTIVATION_SHA"
test "$(shasum -a 256 "$VALIDATOR" | awk '{print $1}')" = "$VALIDATOR_SHA"
python3 "$VALIDATOR" \
  --receipt "$RECEIPT" --expected-receipt-sha256 "$RECEIPT_SHA" \
  --activation "$ACTIVATION" --expected-activation-sha256 "$ACTIVATION_SHA" static
python3 -m pytest -q tests/test_motion_backhand_loop_bc_schema2_fk_consume_activation.py
```

Expected source result is `PASS static ... attempts_started=0 consume_not_run=true`; focused tests
must report `28 passed`. The validator deliberately has no `consume` subcommand. The activation remains
`review_required_not_consumed`: it binds the successful interpreter, exact checkout/tool/plans,
donor, PKL/report lineage, output roots and full commands, but it does not run them.

Only after this branch is reviewed and merged may an operator use the activation's exact argv,
serially. Immediately before each command, recheck the detached source commit/cleanliness, exact
runtime and all input hashes, and require that that asset's output root does not exist. Starting the
command spends that asset's sole authorized attempt. On any rc != 0, preserve evidence and stop that
asset even if cleanup leaves the output root absent; do not automatically retry or advance the
fallback ladder. On success, `schema2_fk_report.json` must be published last. This activation alone
does not authorize L0/L1, table/net, dynamics, simulator, training, formal-motion or hardware work.
No `consume` command in this section was run while creating the receipt/activation change.

A future completed schema-2 NPZ still is not an accepted motion. Its exact report must first be
tracked, then a separate L0 audit decision can be made; vendor L1 self-collision and full-trajectory
table/net clearance remain later gates. Never advance B/C fallback for an internal schema-2/FK
failure.

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
