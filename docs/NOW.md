# NOW — Active Work Board

Short-horizon board: what is being worked on RIGHT NOW, by whom, on which branch, and what the
next checkpoint is. The long-horizon roadmap lives in
[gates/G08_blind_spot_improvements.md](gates/G08_blind_spot_improvements.md); history lives in
[PROGRESS.md](PROGRESS.md).

Rules:

1. **Claim before you code.** Add/update your row here (owner + branch) BEFORE starting a work
   item, in the same push as your branch. This exists because we already built the same feature
   twice (no-teleport wrap: `3eba347` on main vs the `rsi-on-wrap-progress-fix` branch).
2. One row per active item; move finished rows to the Done section with a date, and put the
   substance into PROGRESS.md / the gate doc.
3. Franco arbitrates priority; anyone can edit their own row.

## Team

| Person | Focus |
| --- | --- |
| franco | Direction, priorities, arbitration |
| jiayi | End-to-end training bring-up; reward tuning |
| yikang | Deployment; sim/env alignment |
| claude (franco's agent) | Foundation: infrastructure, A/B experiments, doc/code hygiene |

## Active

| Item | Priority | Owner | Branch | Status / next checkpoint |
| --- | --- | --- | --- | --- |
| P2.1 A/B: no-teleport+stand-start vs teleport-era (2000 it, seed 1) | ★★★ | claude | `p2-multiswing` | RUNNING on pod GPU1/GPU2 (started 2026-07-03); next: scoreboard both arms, post curves here |
| P2.3: target-reference consistency — adaptive tracking sigma (SMASH) + sampling audit | ★★★ | claude | `p2-target-consistency` (to open) | design mapping in progress; next: implement sigma curriculum, A/B vs fixed std |
| A8: post-swing initial-state buffer (Ace recipe, third reset branch) | ★★★ | claude | `p2-multiswing` (follow-up commit) | implementation mapped; next: code + mechanics check |
| P2.0: ready-pose definition (see G08) | ★★ (foundation) | **unassigned — needs franco's source decision** | — | decide source: record a ready-stance video vs handcraft joint pose vs average of clip start frames; then re-process clips to start/end at ready |
| Legacy-task long run `merged_uniform_hopex` (20000 it, task=HOPEPingPong on own branch) | ? | yikang | `rsi-on-wrap-progress-fix` | RUNNING on pod GPU0. ⚠ branch duplicates main's wrap_teleport machinery and has LFS-pointerized CSVs — reconcile with main before merging; the unique progress-fix is already ported to `p2-multiswing` |
| Reward tuning (current focus unknown — jiayi please claim/describe) | ? | jiayi | ? | — |
| G07 mocap→runner bridge + world→robot target transform design (A2) | ★★ | unassigned (natural fit: yikang) | — | design doc first; see G07 Next Steps and G08 audit item 2 |

## Queued (priority order, from G08)

1. A1: target latency / mid-swing re-sampling / obs delay injection (stack on P2.1+P2.3 winner).
2. P2.2-lite: orientation-normalize the existing two clips at retarget (`reground_hope_frame.py`).
3. A5: record 30-50 new reference swing videos (needs a human + camera; processing pipeline ready).
4. P2.5-lite: ball + drag/bounce + PACE at-contact landing reward (independent track; big).
5. A3: per-joint actuator ID on the real A3 (needs hardware time).
6. G06 acceptance numbers for the shipped checkpoint (needs `model_p4_deployparity.onnx` copied to
   `/workspace/shared/models/` or dongc1's machine).

## Done

| Item | Owner | Landed | Where |
| --- | --- | --- | --- |
| Fixed-protocol sim2sim scoreboard (`scoreboard_eval.py`), validated end-to-end on pod | claude | 2026-07-03 | `p2-eval-harness` |
| 4 main-breaking merge casualties fixed (conflict markers; `motion_file` regression; `episode_time_left` probe crash; `play.py` `_wbt_tasks`) | claude | 2026-07-03 | main / `p2-multiswing` |
| `motion:` task-YAML/CLI plumbing for wrap_teleport / stand_start / hold | claude | 2026-07-03 | `p2-multiswing` |
| racket_progress exact-zero on resample (ported from yikang's `c7733db`) | yikang→claude | 2026-07-03 | `p2-multiswing` |
| RunPod multiuser provisioning + smoke suite | yikang (+team) | 2026-07-02 | pod `/workspace` |
| Doc realignment to simtoreal2 reality; Phase 2 roadmap into G08; papers | claude | 2026-07-03 | main |
| First sim-to-real (forehand-only, `model_p4_deployparity.onnx`) | yikang/dongc1 | 2026-07-02 | main |

## Update Rule

Update your row when: you start/finish an item, change branch, hit a blocker, or hand something
off. Keep rows one line; details go in the gate doc or PR description.
