# Legacy retarget solvers

Recovered tools, kept because a bank's receipt pins them by SHA-256 and the bank cannot be
reproduced or re-solved without the exact bytes.

## `solve_chingmu_canonical_racket_full_phase.v4d_20260803.py`

SHA-256 `d6d6bfddb518e3809a1a39ee1fe0779703d8539370b1041ee82e56267f057af5`.

This is the solver that produced `assets/motions/chingmu73_measured_v4_20260803` (its
`COMPLETION_MANIFEST.json` pins exactly this digest). It was never reachable in git history: it
survived only as a dangling blob, `0f03b6284e643f64c79db909b526c38a694c5b70`, from a commit that
was later rebased or amended away. A copy also sits on Pod2 under
`/workspace/codexschema/chingmu_racket_v4d_exact_20260803.kRiC8j/`. Both were byte-verified against
the pinned digest before this file was written. Tracking it here is the point: a dangling object is
one `git gc` away from gone, and Pod storage is not an archive.

**It is not the current solver and must not be used as one.** Relative to the tracked
`../solve_chingmu_canonical_racket_full_phase.py` it is 895 lines against 1260, takes 10 CLI
arguments instead of 13, has no `--urdf` (it never reads a URDF at all), and — the difference that
matters — it has no `constrained_frame_bounds`, so it never checks whether a joint can physically
reach the next frame's target from the current one. That check is why the current solver admits
only 19 of 73 clips where this one admitted 73 of 73; the extra 54 are motions the newer tool
judges mechanically unreachable.

Use it to reproduce or re-solve the v4-lineage bank. Do not use it to certify new motion.
