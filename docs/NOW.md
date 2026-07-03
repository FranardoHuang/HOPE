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
3. Priority ordering is maintained by claude (franco's agent) and discussed with franco; anyone can
   edit their own row.
4. **This file lives on `main` ONLY.** Never edit NOW.md on a feature branch (it would fork the
   board and merge back stale). Claim/update flow from any branch, without switching:
   ```bash
   git fetch origin && git show origin/main:docs/NOW.md   # read the live board
   # edit + push a docs-only commit straight to main:
   git stash -q; git switch main && git pull --ff-only && $EDITOR docs/NOW.md \
     && git commit -am "now: <one line>" && git push && git switch - && git stash pop -q
   ```
   Docs-only commits to main need no PR/review; everything else goes through branches.

## Runtime Estimates (RTX 5090, measured 2026-07-03)

| Job | Cost |
| --- | --- |
| DeployParity training, 4096 envs | ~4.7-5.3 s/iter → **2000 it ≈ 2.8 h · 8000 ≈ 11 h · 12000 ≈ 16 h · 20000 ≈ 27 h** |
| Kit boot + env build (per run) | ~2 min |
| Mechanics check (512 envs, 25 it) | ~3 min |
| ONNX export (play.py) | ~2 min |
| Scoreboard, 4 protocols × 2400 steps (CPU) | ~25 min/checkpoint (parallel to training) |
| Reference lineage | shipped model_p4 ≈ 15-25k it; treat <8k as immature for deploy protocols |

## Team

| Person | Focus |
| --- | --- |
| franco | Direction, priorities, arbitration |
| jiayi | End-to-end training bring-up; reward tuning |
| yikang | Deployment; sim/env alignment |
| claude (franco's agent) | Foundation: infrastructure, A/B experiments, doc/code hygiene |

## Plan To Saturday (2026-07-03 → 07-04, target: play at the venue with an improved policy)

Critical path (GPU 1/2):

1. **Tonight ~19:30** — A-ext/E-ext hit 14k iters → 4-protocol scoreboard verdict (claude, ~1 h).
2. **Tonight** — warm-start the **explicit-clipped-PD fine-tune from the E champion** (the
   pre-hardware gate recipe, ~8000 it ≈ 11 h overnight, GPU1); GPU2 = same ft from A-ext champion
   as backup. (claude)
3. **Saturday morning** — export → MuJoCo explicit gate + deploy-faithful → if pass, build the MDU
   package (`build_a3_deploy_pkg.sh`) per PINGPONG_NEW_CHECKPOINT_TUTORIAL. (claude prepares,
   yikang ships/runs on the MDU)
4. **Saturday afternoon** — deploy & play: forehand first; **try backhand in SHADOW mode** — E was
   trained for stand-entry, this is the potential headline of the day. (yikang + franco)

Parallel tracks (no GPU conflict):

- **Ball physics v1 (P2.5 prerequisite)**: mocap ball-trajectory collection is happening NOW at the
  venue → fit drag/bounce from the fresh recordings (planner calibration path); spin-aware physics
  arrives with the simtoreal2 merge. Suggested owner: jiayi (+claude for pipeline).
- **P2.0 + A5 bundling REMINDER**: while the mocap rig is up, also record the READY-STANCE clip and
  30-50 orientation-normalized swing videos (face the incoming-ball direction!). 0703 teacher-clip
  uploads suggest this may be partly done — confirm coverage before teardown. (franco/on-site)
- simtoreal2 → main merge + doc updates (claude, in progress).

## Active

| Item | Priority | Owner | Branch | Status / next checkpoint |
| --- | --- | --- | --- | --- |
| P2.1/P2.3/A8 ablation ladder (arms A-E) | ★★★ | claude | `p2-multiswing` | 2000-it round DONE (Isaac composite: A 0.42 / B-teleport 0.79 / **C+sigma 0.62** / D+postswing 0.40; adaptive sigma = +48% over A at equal budget). All arms too immature to survive MuJoCo deploy protocol (fall ~1.2 s from stand) — overnight: GPU1 A-ext→12k (control), GPU2 **E-ext = C-resume + post_swing + sigma →12k (product candidate)**; B-ext dropped (hardware already proved teleport-era fails stand-entry). 4-protocol scoreboard verdict when done |
| P2.3: adaptive tracking sigma (SMASH) | ★★★ | claude | `p2-multiswing` (flag `racket.adaptive_sigma`) | IMPLEMENTED + mech-verified 2026-07-03 (sigma live-updates within clamps); next: arm C after P2.1 A/B |
| A8: post-swing initial-state buffer (Ace) | ★★★ | claude | `p2-multiswing` (flag `motion.post_swing_start_prob`) | IMPLEMENTED + mech-verified 2026-07-03; next: arm D after P2.1 A/B |
| P2.0: ready-pose definition (see G08) | ★★ (foundation) | franco (拍摄) + claude (pipeline) | — | DECIDED 2026-07-03: option (a) — record a ready-stance video through GVHMR→GMR on the next site visit (bundle with A5's 30-50 new swing clips); claude processes + wires into stand_start/hold/clip re-entry |
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
