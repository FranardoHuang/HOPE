# G08 Blind-Spot Improvements

Status: Research track (Phase 2 roadmap recorded 2026-07-03)

> **命名提醒：** 本文件的 “Phase 2 Performance Roadmap” 是 2026-07-03 建立的宽泛性能研究桶，
> 不是现行课程“阶段 2”的定义。2026-07-08 已把课程重排为阶段 1 固定点、阶段 2 虚拟球变到达
> 状态、阶段 3 物理球进场；连续恢复和部署验证另列。当前定义与状态只看 [NOW](../NOW.md)。

## Goal

Improve beyond the HITTER-compatible baseline by targeting weaknesses that can decide matches.

The baseline exists as of 2026-07-02: the unified swing policy transferred to the real A3
(forehand only, scripted targets). Phase 2 is about performance; the reference papers in `papers/`
are the evidence base: HITTER (`2508.21043`), PACE (`2509.21690`, the TTRL-ICRA2026 paper), SMASH
(`2604.01158`), Sony Ace (`s41586-026-10338-5`).

## Phase 2 Performance Roadmap (team, 2026-07)

Each item lists the failure being targeted and the paper-backed mechanism to try first.
Priority (franco, 2026-07-03): **P2.3, A8, P2.1 first** (with P2.0 as their shared foundation);
active assignments live in [../NOW.md](../NOW.md).

### P2.0 Ready-pose definition (foundation for P2.1/P2.4/backhand stand-entry)

Missing piece surfaced 2026-07-03: the whole plan assumes a defined 准备动作/ready stance, but none
exists. Today "ready" is implicitly one of two wrong things: the pre-swing hold freezes the
reference at the CLIP'S FIRST FRAME (whatever skewed stance the source video started with — the
P2.2 problem), and `stand_start` initializes from `default_joint_pos` (a neutral robot stand, not
a table-tennis athletic stance). Everything downstream needs the real anchor:

- P2.1 recovery must know what to recover TO;
- P2.4 stitching is "ready → strike → ready" by definition;
- backhand stand-entry and the deploy idle pose are the same stance;
- PACE 把 residual action 锚在举手的 nominal ready pose；SMASH 的 clip 都经过准备/恢复阶段；
  ACE 使用近时优的 reset MPC，并按来球和期望落点选择 prepare reset target。但 ACE 是固定安装、
  非自由站立机器人，不能替代 humanoid 平衡恢复或挥拍中途随机来球的证据。

Plan (v0 needs a source decision — franco): (a) record one short ready-stance video through the
existing GVHMR→GMR pipeline (cheapest, consistent), (b) handcraft the joint pose, or (c) average
the orientation-normalized clip start frames. Then wire it in three places: the `stand_start`
branch initializes to READY (not default stand), the pre-swing hold holds READY (not clip frame
0), and the clips are re-processed to enter/exit through ready (P2.2-lite pass).

DECISION REVISED (franco 2026-07-04 evening): no dedicated ready video. The v5 clips
(`hope_{forehand,backhand}_v5.npz`) start AND end near the ready stance — measured: the two
clips' first frames agree to 0.15 rad mean joint distance (a usable shared ready anchor), and
each clip's last frame is 0.24-0.27 rad mean from its first (RL learns the in-between filling;
the clip_switch/post_swing/hold machinery already trains exactly those transitions). So P2.0
reduces to option (c)-adjacent at zero capture cost: extract the ready reference from the v5
first frames and wire it into `stand_start`/hold. The 6-anchor capture spec
(`../motion_and_contract_v3.md` §4) still records every future clip ready→…→ready, which keeps
this property by construction.

### P2.1 Balance across consecutive swings

Failure: the robot falls after several swings; it does not recover weight/posture into a robust
ready stance. Status correction (2026-07-03): the structural machinery already exists on main
(`3eba347`) — 10 s multi-swing episodes, `wrap_teleport=False` (no teleport between swings),
`stand_start_prob=0.25` (deploy-entry resets), pre-swing hold — but no shipped checkpoint was
trained with it (the deployed model predates it, hence the teleport-entry backhand gap). The task
is therefore to train and validate under this structure (P2.1 A/B on branch `p2-multiswing`), not
to build it.

- HITTER: 10 s episodes chain swings; after each swing the next swing type and targets are
  resampled; the base-position reward is active only pre-strike, which *is* the ready phase
  (Sec V-B). Restoring multi-swing episodes is the primary fix.
- PACE: up to 5 consecutive serves per episode + fall termination + residual actions anchored to a
  nominal ready pose → recovery emerges without a dedicated reward.
- SMASH: every reference clip embeds a ~0.54 s recovery segment after contact plus a cyclic strike
  phase variable, making cycles chainable.
- ACE（可借鉴但不能直接外推）：训练单拍时，可从早先训练 episode 保存的动态 reset plan 中采样
  初始状态，而不是总从 stationary default state 开始。

### P2.2 Reference-motion orientation normalization

Failure: collected swing videos were not rotated to face the incoming-ball direction, so imitated
stance and swing are skewed.

- HITTER: base orientation command is always "face forward"; references are anchored
  pelvis-relative — heading is never inherited from the video.
- SMASH: strike-target features and motion matching are anchor-relative; the racket-orientation
  objective uses `|n·v̂|` (driven by commanded strike velocity, not the clip's facing).
- Action: normalize orientation at the GMR retarget/anchor step and keep the base-heading command
  fixed forward.

### P2.3 Target distribution vs teacher motion consistency

Failure: the sampled target's center (hit position, racket velocity) differs from what the teacher
motion represents, forcing the policy to learn two things at once.

- HITTER: targets are sampled conditioned on swing type in non-overlapping per-swing regions;
  workspace coverage comes from moving the base command, not widening the racket-target box.
- Ace: hindsight experience replay — relabel each transition with the achieved state as the target
  (removes the two-objectives conflict at near-zero cost) + event-table stratified replay to keep
  rare high-quality strikes represented.
- SMASH (systematic solution): expand the clip library (Motion-VAE 400 → 5k), retrieve the
  nearest-reference per sampled target, region-adaptive sampling, and adaptive tracking sigma
  (their ablation: removing adaptive sigma collapses success 86.4 → 22.6).
- PACE: tolerance-clamped tracking rewards (stop rewarding once inside tolerance) preserve
  exploration freedom around the target.

Relayed SMASH-author conversation (二手转述 via team member, 2026-07-03, 未验证 — weigh below
the paper), affecting P2.3/P2.4 design choices:

- **Strike region**: torso-frame plane anchoring with local refinement — CONSISTENT with the
  paper's anchor-relative target features (Sec IV.A), adopt as a design reference for
  base-relative targets. ⚠ The relayed claim that y/z caps existed "mainly to reject
  egocentric-perception outliers" CONFLICTS with the paper's own framing of the strike volume as
  the reachable hitting workspace (Fig. 5) — treat as unverified. Our y/z widening decisions
  follow our own measured reachability + the footwork curriculum, not this rationale.
- **Clip length is not a magic number**: their ~1.08 s strike segments (0.54 s each side of
  contact) are a scale for THEIR data; segment length must be re-calibrated to our own strike
  timing. Note OUR clips are 2.6-2.8 s (139/132 frames @ 50 Hz) — 2.5× longer than theirs. A
  **clip-trimming experiment** (cut a ~1.2-1.6 s window centered on the detected strike phase) is a
  cheap candidate: tighter strike-centered imitation, more swings per episode, less irrelevant
  follow-through imitation. Queue after the current ablation round.

### P2.4 Locomotion aggressiveness / ready motion / clip stitching

Failure: the robot moves too violently toward targets; no preparation motion; needs
target-conditioned stitching/scaling of ready+strike references and eventually rallies.

- HITTER: position-style base commands (with fixed forward heading) produced faster AND calmer
  single-step reaches than velocity commands. Note: the deploy-parity redesign removed the base
  target/reward entirely — with the mocap base pose available at play time, re-adding the base
  command mechanism (as an ablation at minimum) is paper-supported.
- PACE: pseudo-velocity command proportional to remaining position error → smooth deceleration
  into the strike stance; predictor-based proactive targets (reactive control is the root cause of
  late/violent motion).
- SMASH: autoregressive Motion-VAE generates the next reference segment from the robot's *current*
  state (built-in stitching); strike rewards gated to short windows around impact avoid sharp wrist
  accelerations; phase-dependent target noise trains tolerance to late target updates.

Status (2026-07-04): two v1 pieces landed on main as default-off flags — `motion.clip_switch_prob`
(deploy-parity mid-swing clip switch; venue 准备/正手/反手-switch falls root cause) and the
PACE-style `base_decel` reward (`rewards.base_decel_weight`, pre-strike pseudo-speed tracking).
Signal-tier arms R11/R12 in NOW.md. The v2 deceleration design (fitted accel/decel envelope +
direction + time budget + stroke-amplitude coupling — franco's correction to the PACE P-law) and
the continuous-intensity motion library q_ref(φ,ρ) are specified in
[../motion_and_contract_v3.md](../motion_and_contract_v3.md).

### P2.5 Ball-flight physics modeling (landing point, spin, ball-quality rewards)

Failure: no ball model at hit time → cannot reward landing point, pace, angle, or spin; cannot
switch strategy objective (win the point vs sustain the rally). Note: the trained task currently
contains NO ball at all; ball spin is not yet measured by the rig (a patterned/rigid-body ball and
a relay change are prerequisites — see `docs/interfaces/frames_and_coordinates.md`).

- HITTER: drag+gravity flight + diagonal restitution bounce, identified from 15 trajectories; its
  Eq. 5-6 run forward from racket state at contact predict landing/pace — directly a reward.
- PACE (the key reward template): at the instant of paddle contact, physics-predict the landing
  point and net-crossing height and reward them immediately; their ablation shows sparse
  landed/not-landed feedback alone never learns returning.
- SMASH: AEKF with bounce handling (removing bounce handling explodes prediction error
  3.5 → 12.7 cm); analytic inversion of a linearized drag model — chosen flight time T_f is the
  pace knob. **AEKF de-scoped for our mocap setup** — primary evidence: HITTER
  itself used plain 31-sample polynomial fitting at 360 Hz mocap with no Kalman filter and hit
  92.3% real return rate (Sec IV-A); corroborated by a relayed SMASH-author conversation
  (二手转述, 2026-07-03, 未验证) saying their AEKF mainly served 60 Hz egocentric noise. FINAL
  arbiter = our own fit residuals on the fresh trajectory recordings — if residuals are clean,
  polyfit stands; keep BOUNCE SEGMENTATION regardless (model structure, not filtering).
- Ace (ceiling): velocity-dependent Magnus model, data-fit table/racket contact + residual MLP,
  spin measured at 400-700 Hz; policy-sampler modes are exactly the win-vs-rally objective switch.
- In-repo head start: `hope_planner` already has drag+bounce with traj01 calibration, the
  `tasks/table_tennis` Isaac scene has drag/Magnus hooks, and the unmerged branch
  `ball-physics-realistic` (commit `0098c43`) contains mocap-calibrated spin-aware ball physics
  fitted from 300 Hz recordings.

### P2.6 Smash (on top of P2.1-P2.5)

- SMASH's lesson: train smash as a SEPARATE policy with dedicated smash reference data (longer
  execution horizon than the shared strike policy handles). Their smash deployment was limited by
  egocentric-camera FOV during aggressive postures — our external mocap does not have that
  weakness. Relayed author conversation (二手转述, 2026-07-03) confirms what the paper already states
  (p.11): main policy contains no smash; demo smashes are a separately trained policy; adds that
  systematic integration was attempted and dropped for difficulty/deadline. The open extension they
  themselves point to — and our concrete P2.6 design — is a **policy-selection mechanism** choosing
  between normal-return and smash policies per incoming ball (cf. Ace's policy sampler).
- Pace maps analytically: shorter chosen flight time → larger commanded racket velocity
  (HITTER Eq. 5-6 / SMASH Eq. 28-29); the WBC already tracks commanded racket velocity, so smash
  training is target-velocity-distribution + dedicated references + P2.5 rewards.

## Audit-Derived Additions (2026-07-03)

High-leverage items the papers treat as first-class but the roadmap above does not cover:

1. **Latency and target-time-variance modeling.** Training currently assumes a perfect, constant
   target from t=0; the real loop delivers late, jittery, converging targets (SMASH quantifies
   ~10 cm error at 0.6 s-to-strike → ~1 cm at contact and injects tts-decaying target noise; PACE
   injects delays/noise at real-sensor magnitudes; Ace models latency/dropout). Measure end-to-end
   mocap→tick latency and train with target delay + mid-swing re-sampling before the mocap bridge
   lands.
2. **Planner→policy frame-transform ownership.** The 175-D actor is base-position-free, but
   converting a HOPE-world planner target into the robot-relative frame requires the mocap base
   pose at the interface boundary (plus the engage-time yaw alignment). This transform currently
   has no owner and is masked by scripted targets. Design it before/with the mocap bridge (G07
   Next Steps).
3. **Actuator system identification.** PACE's sim-to-real gap (94 → 61%) came from one
   mis-modeled transmission; our implicit/explicit-PD incident is the same class. Per-joint
   step/chirp ID (arms + waist first), fit delay/bandwidth (Ace uses delayed-LTI joints), feed
   both sims. The elbow-at-6.7× -torque-limit trace shows the policy currently exploits actuator
   fictions.
4. **Evaluation infrastructure + real-data flywheel.** Fixed serve corpus (recorded real serves
   replayed in sim), fixed target grids, per-checkpoint scoreboard on `mujoco_eval_onnx.py`,
   mandatory full mocap/obs/action logging for every hardware session (the MDU captures already
   found the yaw drift and the elbow saturation — make it policy), automatic hit detection from
   ball-velocity discontinuities for real hit/return/landing stats. Partially realized 2026-07-04:
   venue-ball-driven eval (`--target-source venue-balls`, see G06) covers the "real incoming
   distribution" half; the fixed-serve-corpus replay half stays open.
5. **Reference-motion scale.** 2 clips vs SMASH's 400(+5k generated). The GVHMR→GMR pipeline
   already works; 30-50 orientation-normalized clips across the target workspace is the cheapest
   attack on P2.2/P2.3. MVAE generation can come later (SMASH: 5k→10k was already marginal).
6. **Fall management as a subsystem.** Guard thresholds must be derived from the trained state
   envelope (the 0.6 rad squat-guard incident), take-over should damp rather than snap, and a
   recover-to-ready behavior is the long-term answer (Ace always keeps a verified-safe reset
   trajectory). Fewer falls = more data per hardware session.

## Candidate Tracks (longer-term)

1. Spin perception and spin-aware ball dynamics (P2.5 extension; `ball-physics-realistic` branch).
2. Double-bounce and short-ball handling.
3. Deep-ball and non-fixed hitting-plane handling.
4. Serve generation (Ace: demonstrated toss stitched to an optimized strike).
5. Topspin, backspin, sidespin, block, chop, and loop stroke repertoire.
6. Opponent intent modeling and tactical adaptation (Ace policy sampler; HITTER cites Latte-MV).
7. Multi-agent training.
8. Vision-based perception to reduce mocap dependency (SMASH is the humanoid existence proof).

## Related Directories

- `hope_ws/src/hope_planner`
- `hope_training/whole_body_tracking`
- `agi/A3_MuJoCo_Sim/`
- `external_repos/TTRL-ICRA2026` (PACE's public training code)
- `papers/` — HITTER, PACE, SMASH, Ace
- branch `ball-physics-realistic` (spin-aware ball physics, unmerged)

## Operation Docs

Pick the operation doc based on the selected mini-spec. Common starting points:

- [../operations/run_planner.md](../operations/run_planner.md)
- [../operations/run_training.md](../operations/run_training.md)
- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)

## Acceptance Criteria

Each track needs its own mini-spec before implementation:

- Failure case being targeted.
- Measurable improvement metric.
- Required data.
- Required simulator changes.
- Deployment risk.
- Owner and expected demo.

## Current State

Done:

- HITTER paper limitations identified: fixed hitting plane, external mocap dependency, ignored
  spin, limited stroke repertoire, no autonomous serving, no opponent adaptation.
- All four reference papers read and mapped to the roadmap (2026-07-03, see above).
- Baseline exists: first sim-to-real of the unified swing policy (2026-07-02, forehand only).
- TTRL/PACE code locally available via `scripts/sync_external_repos.sh`;
  `HOPE-TableTennis-AgibotA3-v0` provides the candidate Isaac scene for ball/serve experiments;
  `ball-physics-realistic` branch holds spin-aware ball physics.
- A first P2.1/P2.4 mini-spec now covers private Franco/v6/v7 motion intake,
  native-vs-TOPP timing, two-vs-four action selection and arbitrary-time
  strike/absorb/recover/ready transitions; see the 2026-07-11 section below.

Not done:

- The new P2.1/P2.4 mini-spec is not simulator-accepted yet. All ten videos now
  have canonical-beta GMR, content-bound root-z grounding and a dense 240 Hz
  sampled ground/self/body-clearance screen: 5,162 samples show no ground
  danger, self-collision, `<5 mm` body clearance failure or `<20 mm` warning.
  This closes the earlier `7.7--8.4 cm` floor-penetration finding; it does not
  promote a motion. The frozen canonical-frame 64-question screen returned
  zero common support, so two-vs-four remains inconclusive. Exact schema-2
  candidate materialization, spatial strike-point retarget, whole-path
  table/net clearance, vendor L1 dynamics and Gate3/Gate3B remain open.
- The audit-derived items 1-4 are not yet scheduled; item 2 must land before/with the mocap
  bridge.

## Risks

- Stacking new rewards before restoring the two dropped HITTER structures (multi-swing episodes,
  target-reference consistency) treats symptoms; fix structure first.
- The highest-impact blind spot may shift once mocap-in-the-loop play produces real failure modes.
- TTRL may drift upstream; mini-specs must record the TTRL commit when extracting ideas or code.

## Next Steps

1. Implement the P2.1 mini-spec's fail-loud preprocessing and ready-set
   measurement gates before launching recovery RL.
2. Schedule audit item 2 (frame-transform design) into the G07 mocap-bridge work.
3. Start the P2.5 prerequisite chain: patterned ball + relay orientation forwarding + serve-corpus
   recording, and evaluate the `ball-physics-realistic` branch for merging.
4. Run `scripts/sync_external_repos.sh` before using TTRL/PACE as a reference and record the
   commit.

## V5 professional-transfer and Phase accelerator (2026-07-10)

这条短期研究线的人类责任人是 franco，执行者是 Codex。要回答的问题是：
专业人类的路径、接触几何和从近端到远端的发力顺序，能否成为 A3 有用的软先验；
不是要证明 A3 能逐关节照抄人类动作。

The preregistered axes are:

1. task-only vs V4 soft teacher vs V5 professional soft teacher;
2. human timing vs A3-retimed timing;
3. original path vs guarded backswing extension (+20/+40%), follow-through
   rewrite and their combinations;
4. historical site/ball-centre co-location vs versioned exact face contact;
5. contact-frame and `+-1/+-2` velocity-window definition;
6. matched task strike speed around 2.2 vs 3.4 m/s, holding path and face fixed;
7. a common external question bank for task-only, V4 and V5 teachers;
8. legacy incoming box vs venue-rebalanced training and matchlike exam.

Path extensions enter GPU training only if offline evidence proves the
original path is limiting and shows positive stroke-length, lower `a_min` and
better torque margin. The accelerator materializes a content-addressed
manifest, rejects hard-guard failures before training, uses a 512-env mechanism
smoke, paired conservative halving at 4096 envs, and sends at most two recipes
to a three-seed mature run. Research promotion is 50% all-attempt return rate;
hardware candidacy is 80% plus safety/deployment gates.

See
[v5_professional_transfer_audit_2026-07-10.md](../research/v5_professional_transfer_audit_2026-07-10.md)
and `scripts/v5_ablation_accelerator.py`. Phase follows the V5 limit study, but
does not wait for perfect imitation: it advances with the corrected ruler and
uses task-driven lower body, per-question timing and position/spin curricula.

The 2026-07-11 local-work audit also retained three causal follow-ups in the
current feature/experiment registry, without authorizing a training launch: S1 old-helper/S1-only/
S1+guidance continuations from the same checkpoint and budget; venue-shaped
temporally correlated prediction error versus a shuffled control and an
explicit confidence/history input; and separation of R8's envelope penalty
from its RSI stand-height change.  These are paired ablations, not requests to
complete every historical report cell.  The schema-v3 canary and
`noise_scale=0` shortlist come first.

## Motion library, TOPP and arbitrary-time recovery mini-spec (2026-07-11)

The scoped mini-spec is
[motion_library_topp_recovery_2026-07-11.md](../research/motion_library_topp_recovery_2026-07-11.md).
It records the exact ten-video intake, the high-risk self-collision regions,
air-swing strike-phase limitation, native/TOPP paired contract, fair two-vs-
four experiment, stable per-question selector and event-driven recovery
state machine. It also distinguishes three backhand-loop recordings as one
candidate group and keeps Franco/v6/v7 as separate ready-pose families until
measured transition gates pass.

The execution order is deliberately front-loaded with cheap falsification:
content audit -> GVHMR/GMR/schema-2 -> L0/self-collision/table-net clearance ->
returnability phase scan -> TOPP and repeat gates -> dynamic clip catalog ->
paired training -> T0/T1 recovery. The memory-gated Pod1 GVHMR queue completed
10/10 structural reconstructions with full bindings tracked in
`configs/motion_video_gvhmr_results_20260711.json`. A separate CPU-only,
bundle-bound GMR queue then completed 10/10 finite 30 Hz, 31-DoF diagnostic
  retargets; `configs/motion_video_gmr_results_20260711.json` records all
  input/output/log/audit hashes and keeps `formal_eligible=false`. Later
  canonical-beta grounding and dense safety evidence closes the discrete floor
  penetration and sampled body-clearance prerequisites, as summarized above,
  but there is still no promoted exact schema-2 clip, four-action actor,
  recovery policy or hardware candidate. The mini-spec does not authorize
  real-robot testing.

Evidence immutability was rechecked on 2026-07-12. The original per-video-beta
GMR result continues to bind the exact auditor SHA used to generate it; the
later canonical-beta auditor extension is not retroactively substituted into
that ledger. Intake validation now reports path escape, unsafe output stem and
wrong extension as separate fail-closed errors, including dot-only names such
as `..mp4`. These are provenance/diagnostic repairs only: no motion was
promoted, no Pod job was launched and no hardware action was authorized.

## 恢复与等待的原文审计结论（2026-07-13）

这轮审计针对“上一拍尚未完全结束时，下一拍在任意时刻到来”这个具体问题逐项核对原文，
不再把论文名当作笼统背书。ACE 支持可打断的近时优 reset bridge 和条件化 prepare pose，
但没有自由站立 humanoid 的平衡债；HITTER 只在一拍完成后换题；SMASH 的 preparation/recovery
半段、循环 phase 和边界平滑服务于离线扩库及 runtime motion matching；PACE 的五连发也不是
挥拍中途随机揭题。因此这些工作都不能直接证明 A3 的任意时刻恢复或 vendor MuJoCo Gate3B。

现行设计把硬安全、动态平衡债、ready-set potential 和随机到球拆开。随机到球首先是
环境/题目/截止时间轴，由真实下一拍评分，不是第三项 dense reward。T0 保持按完整周期换题；
T1 只改成事件驱动结构并冻结 reward；T2 才允许 learned shaping。若 T1 仍失败，先用配对 seed
做平衡债与 ready potential 的 `2^2`。只有一个不偷看未揭示题目的 readiness critic 在独立
训练/校准集上锁定，并一次性通过与 sealed Gate3B q50 隔离的 critic-gate q50，才能扩成 `2^3`。
存活三项再做七点固定总预算 simplex，并至少补一个第二总预算量级；只固定总量无法识别 PPO
对 reward 绝对尺度的敏感性。完整原文、T0/T1/T2、非类比边界和 q50 失败条件见
[连续挥拍时序审计](../research/phase1_continuous_rally_timing_2026-07-11.md)。这只是设计和文献证据，
不增加 G05/G06 runtime credit，也不授权 Pod、simulator 或真机运行。现有 machine prereg、
validator 和 operation 仍固定旧三 reward/full `2^3`；新设计尚未物化，必须另建内容寻址版本，
不得改写旧证据或按旧合同点火。

动作库也采用同一条“不能靠类比晋级”的规则。冻结站位得到 `0/64`，只说明旧固定题没有在
同一帧覆盖这些动作，不表示动作本身无效。有效性必须同时满足：动作自己的安全触球时空流形、
适配的来球/动作题族，以及合法的整轨 `SE(2)` 站位。反手拉 B（frame 49，intrinsic `32/32`，
最近旧题 `0.165 m`）和 C（frame 50，`27/32`，`0.237 m`）仍只是满足 `0.30 m` 平移范数粗界的
候选；正手先解决约 `170°` 拍面符号歧义，挡球必须另出挡球题。schema-2、L0、vendor L1
self-hit 和整轨桌网扫掠余量 `>=5 mm` 只授予训练资格，不证明回球有效。

2026-07-13 的 `n/-n` 负控已证明旧动作 virtual-return scorer 对物理拍面符号失明。未执行的
spatial-retarget prereg 因此从 `d8c918ac...5a9f` 重绑到 `0f757c8c...af66a`，要求 raw-A、每侧
`[+1,-1]` physical-B 与 pre-orient +X/hemisphere 门；这只修量尺。已完成的 v5 保留安全/frame
子树，但回球/phase/library 列降为旧 unsigned-plane 诊断，不追认、不覆盖。修复后真实 v5 screen 已产生
B/C `19/3` 个 bounded proposal，但全部缺候选证书，因此仍没有新动作晋级。
真实 v5 输入首次点火又发现 contract 字段读取层级错误：外参未观测声明位于 `frame_contract`，不是
仅含路径/SHA 的 `frame_contract_evidence`。修正后的 tool `d053dd50...5259b` 同时验证 evidence SHA
和显式 false；缺失/true 仍 fail closed。该修复只解除运行前假拒绝，不增加动作行为 credit。

## 组合动作新增设计（2026-07-13）

新登记的 v12/高点拍压/横移动作给长期路线增加三条彼此独立的问题：v12 挡球是否在挡球专卷上
胜过旧候选；高点拍压第五动作是否覆盖四动作库漏掉的高球；一个按横移距离条件化的下肢老师能否
与多个上半身挥拍组合。素材标签只是人类假设，不是已测性能。

横移动作先按准备、击球支撑和恢复三个事件分段；根平移/朝向和腿/足接触由下肢老师负责，上半身
动作以骨盆相对坐标负责球拍目标，骨盆高度/倾角与躯干属于耦合变量。组合必须用受约束的全身求解
保留有符号击球几何、支撑足和安全站距；不能直接“补到同长度后拼关节”，也不能假设左右镜像天然
有效。TOPP 只在几何与接触先过门后用于重定时。

动作选择仍按任务覆盖裁决：每个动作先在自己的来球题族和整轨安全门中合格，再用训练集拟合选择器，
最后在不可变留出集上考试。非击球左臂模仿解除也是单独的配对消融，硬安全始终开启。这些都只是
设计记录，不是动作采用或训练授权。

动作执行顺序已纠正为 Franco 主线优先：2026-07-11 的六段 Franco 动作已有 exact GVHMR/GMR，F0 不
重跑；反手拉 B/C 新增 frame 49/50 的内容绑定名义视觉锚点，但空挥 `contact_truth` 仍为空。F1/F2 的
动作专属题族、`SE(2)`、schema-2 和动力学证书另行闭环，其他三类动作不能被不匹配的旧共用题淘汰。

七段新视频的第一步仍只授权 GVHMR-only，但拆为互不阻塞的 [S0/M0](../DEFINITIONS.md) 离线结构批：S0 仅高点拍压一条，M0 仅四条
横移候选；v12 只作后续 Jiayi 路线对照，本轮不授权。两批各自绑定 nominal 空挥窗口、clean GVHMR
commit、完整权重树、Python、`nvidia-smi`、validator/argv、独立 staging/record/state/output 和唯一
结构 auditor。队列只审本批 source，拒绝 symlink parent，原子 claim output namespace，并从已核验 fd
创建私有只读快照；child 只消费快照，随后复核快照 inode/mtime/ctime/SHA。一批失败不阻塞另一批。
07-11 旧 launcher 仅保留为 gzip 历史源码证据，不是当前可执行入口。横移的机器人后续合同仍以首个准备窗口、朝向对齐后的
双脚分离向量（含前后错位）为终点，不接受更窄合脚替代；GVHMR 本身不声称验证该条件。完整合同见
[GVHMR 预注册实验](../experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md)和
[操作文档](../operations/run_motion_video_gvhmr_prereg.md)。随后 S0 `88/88` 帧、M0 四条
`105/105、97/97、82/82、96/96` 帧已在 Pod1 通过 exact finite structural audit，详见
`configs/motion_video_gvhmr_s0_m0_results_20260713.json`。canonical-beta 后来已完成，但尚无 GMR、
schema-2、足接触/末态脚距、simulator、RL 或真机结果，所以 G08 状态不变，也没有动作晋级。

这五条结果的下一层已收成两份 exact post-GVHMR handoff，并完成 runtime no-clobber consume；S0/M0
handoff exact SHA 分别为 `d57a93e0...a1054` / `60c55150...088ef`。其后的 canonical-beta 也已拆成两份
独立 prereg：只允许注入旧 exact donor，禁止重新聚合新五条；host 新旧专项 `15 passed, 1 skipped`，
真实一加四条 PT 已在绑定 Pod1 CPU runtime 完成 `inspect/consume`，completion manifest SHA 为
`964a7333...f1be3` / `5cef05f7...71a65`，non-beta 内容全 bit-exact。S0 的
`contact_truth`/效果继续为空且禁止借用拉球题；M0 的末态约束明确为去除公共 root、对齐 heading 后的
`right_foot_xy - left_foot_xy` 初始/终态稳健向量，横向站距与前后错位都要保留，脚并拢不算成功。
foot-site mapping 和数值容差须在 GMR result acceptance 前另行预注册。详见
[post-GVHMR 卷宗](../experiments/motion_post_gvhmr_s0_m0_handoff_20260713.md)。
canonical-beta 的计划、运行边界与未来 A3 脚距 null contract 见
[canonical-beta 卷宗](../experiments/motion_canonical_beta_s0_m0_20260713.md)。

## 文档路由更新（2026-07-12）

G08 仍是长期 blind-spot 路线图。当前已采用 setting、阶段/小目标构成和 feature 决定统一放在
[`docs/NOW.md`](../NOW.md)；动作/动作库/恢复实验的详细记录放在
[`docs/experiments/`](../experiments/README.md)。本次文档修改没有运行模拟、训练或真机测试，
也不改变 G08 的 gate 状态。
