# v12, high-press and lateral-motion video intake

Status: Complete
Human owner: Franco
Executor: Codex
Branch: Franco_codex/new-motion-batch-20260713

## Question and decision scope

Verify that the seven newly recorded private videos under `${HOME}/Downloads/{v12,static,motion}`
are present with exact bytes and media properties, and record their intended roles without
claiming that any motion is safe, effective or ready for training.

## Inputs and immutable bindings

The metadata-only manifest is
`configs/motion_video_intake_20260713.json` (SHA-256
`44b00b3c46c837d797990bc6f6255055c0ff83c1bb8643ca81f9707033ca304c`). It binds seven
HEVC 1920x1080 30 Hz files, 82--105 frames each:

- v12 forehand/backhand block: primary v-series candidates requested for Jiayi's mainline;
- `static/pai.mp4`: proposed right-handed backhand high-press fifth action for high contact points;
- two left and two right lateral-step candidates: lower-body locomotion-teacher inputs.

The raw videos remain private and local-only. Their individual sizes and SHA-256 values live in
the manifest; this record intentionally does not duplicate the full table.
The accepting auditor SHA-256 is
`ffdae64ac3437a3d962eb006eadc9d4d429c4a14e41484c6ef9a594b596fc299`.

## Design and controls

Schema 2 separates a stroke video from a lateral-locomotion teacher. A locomotion teacher has no
forehand/backhand or stroke label and cannot be silently consumed as a block clip. The existing
schema-1 Franco/v6/v7 manifest remains valid and byte-compatible.

The labels encode user-supplied hypotheses, not measured performance. In particular, `v12` is not
called the best v-series motion until it passes a task-matched paper, and the high-press action is
not scored on a loop/block paper.

## Acceptance and failure rules

Intake passes only if every relative path stays under the supplied root and the byte count,
SHA-256 and ffprobe fields match. Duplicate JSON keys, NaN/Infinity, unsafe paths, role/action
mismatches and incomplete candidate ranks fail closed. Intake grants no compute, simulator,
deployment or hardware authorization.

## Reproduction

```bash
python3 scripts/audit_motion_video_intake.py \
  --manifest configs/motion_video_intake_20260713.json \
  --source-root /Users/Franco/Downloads
python3 -m pytest -q tests/test_audit_motion_video_intake.py
```

## Results

On 2026-07-13 all seven local files passed exact byte/hash/media validation. The focused suite
passed `11` tests; the repository suite passed `472`, with `9` skipped. No video was copied to a
Pod and no GVHMR, GMR, simulator, RL or hardware
process ran.

## Limitations and claims not made

There is no ball-contact truth, strike frame, mirror/frame proof, schema-2 robot motion, table/net
clearance, dynamics, balance or vendor-MuJoCo result. Air-video semantics cannot establish which
incoming balls a stroke can return.

## Decision and next action

Keep v12 first in the future preprocessing queue, then the high-press action and lateral candidates,
but activate none until the current higher-priority q50/planner/face-sign closures finish and the
offline motion gates have a reviewed consumer. Detailed experiment design is in
`motion_v12_high_press_lateral_teacher_20260713.md`.
