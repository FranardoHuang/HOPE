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

这 22 条 proposal 的候选选择现已机械闭合，但证书门仍未闭合。确定性 consumer 在 B/C 名义 frame 窗内
只合并 3 组 `yaw=0` 的 R0/R1 逐字段相同别名，然后按平移范数、偏航绝对值、回球余量、身体余隙、
frame、candidate ID 冻结顺序。主选为 B `98e7b883...f3c14`（16 个唯一候选中的 rank 0）和 C
`aa0c86fd...f299`（3 个中的 rank 0）。只有桌/网外部几何失败可沿 frozen ladder 前进；schema-2、L0、
vendor L1 或内部动力学失败必须停止该资产。host 专项 `13 passed`，全仓回归
`646 passed, 9 skipped`。没有 GMR、schema-2、simulator、训练或真机行为，因此 G08 状态仍为
Research track，主选也不是晋级动作。完整 ledger 与命令见
[空间重定位实验](../experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)和
[操作文档](../operations/run_motion_spatial_retarget_screen.md)。

## 组合动作新增设计（2026-07-13）

2026-07-17 增加了一个仍保持训练关闭的 ready-to-strike host candidate gate：输入必须是 canonical
schema-2，等待姿态固定为所选 `ready-source` 第0帧且速度显式全零；quintic 只改 join 前路径，触球前
0.1秒六通道逐字节保留。红队用合法 `q/-q` 输入抓到并修复了 join 四元数分量跳变，也纠正了
`joint_vel` 在 join 重算却被误写成全 suffix bitwise 的声明。首次 Pod2 真实生成又抓到标准 migration
三元组被 source gate 假拒绝：两侧均未写候选、未启动 TOPP，不能算动作失败。修复后只接受 core-only
或完整 canonical-v2 三元组，逐位保留 primary source 血缘且拒绝残缺/未知/坏值，专项 `21 passed`。
第二次新 namespace 已成功生成两侧真实 host 候选；production-FK TOPP hard acceptance 全过，但可行
run-up 上界只有正手 `0.64 s`、反手 `0.94 s`，未达 0.5 秒。下一轮已冻结 `6/12/17` 主梯、`9/14`
按需细化和两种 frame0 ready 的端点因子阵，不重放已跑格。Stage 1 raw 数值显示 `d=17` 全面更慢，且
反手 own-ready 改善、正手 own-ready 更优，构成 side crossover；一次性 summary 的 consumer/parser
假绿已由独立 historical attestor 关闭。Pod2 唯一 no-clobber receipt `7cf1c7c9…c377f` 全量重验六格
candidate/schema-2/production-FK TOPP/provenance，六个端点时间 `1.28/0.70/1.54/1.94/0.78/1.42 s`
均为可信 screening evidence、也均未过 0.5 秒。四个 `d=12` 中点的剩余前置仅为 exact tracked
CPU-only runner activation。tracked runner 已通过独立红队并绑定 receipt、冻结输入、唯一 namespace、
contact/fps timing 一致性与 child timeout；generator 从 Stage-1 receipt 认证副本读取，旧 runtime checkout
不再被错误要求含后置 source。v1 四格 generator 已成功但因 candidate float32/TOPP float64 梯度量尺
混用而在 TOPP 前 fail closed；旧 summary `f92e6b8b…63c0e` 冻结。v2 只修量尺并绑定旧失败；唯一
execute 已证明四份 candidate 与 v1 逐字节一致，四个 TOPP 均 rc1、无 timing，summary
`6910db28…f1476` 冻结且不得重放；这些是 v2 的全部正式结论，诊断 `run.log` 不作根因或授权输入。
v3 不再调用 generator，只复验并冻结同四份 candidate，并从 runtime commit 的 Git objects 提供新的
`1 XML + 74 mesh` 完整执行环境：固定
`75 files / 14,127,373 bytes / manifest e0381752…b962de`，拒绝 include、路径逃逸、symlink 和 worktree
fallback。v3 唯一远端 dry-run 在结果 root 前又发现 expected contract 仍误绑 v1，而实际 v2 contract
因 snapshot path 证据不同已有新 SHA；execute/TOPP 未启动。v4 仅修四个 actual contract 绑定并使用新
activation/namespace，相关回归 `68 passed` 且独立红队 GO；随后 dry-run 又在结果 root 前证明 exact log
SHA 后的 `.stl`/英文 errno 文本猜测会假拒绝真实日志，execute/TOPP 未启动。v5 删除重复文本解释但仍
保留日志 SHA，随后因一份 SHA 手抄一字符错误再次 pre-root fail closed。main `8b371eb7` 的 v6 移除旧
V1 summary、generator 副本与全部日志前置，只复验 v2 四份 candidate/contract，保持零 generator 调用，
并在上述 Git-object 闭包中完成唯一远端 dry-run/execute。dry-run 全绿；terminal summary=`b5209bc7…`，
四格 generator=`0`、TOPP rc=`1`、无 timing，`75 files/74 mesh` closure 与无残留正确。只读 forensics
确认四日志同 SHA `f1d5088e…`，共同首错是 `/usr/bin/python3` 缺 `mujoco`，所以它是 runtime dependency
closure 失败、不是动作失败。项目实际 TOPP 依赖仅 `numpy+mujoco`，`scipy` 过度硬门已废除；隔离 probe
证明 `/workspace/hope_mjeval_venv/bin/python` 可加载 `numpy 2.5`、`mujoco 3.10` 与 exact MJCF
(`nq=38,nv=37,nbody=33,ngeom=79,nmesh=74`)。v7 绑定该解释器、完整 package/native dependency closure、
canonical `ldd/readelf` 与 MJCF pre/post snapshot，但唯一远端 dry-run 把 venv symlink 的 `readlink()`
字面 target 误当身份，在 binary/包闭包核验前便因文本不同拒绝；实际 binary 是否漂移仍未知，
execute=`0`。v8 以 canonical realpath、
binary SHA、Python version、venv prefix 和 RECORD/ELF closure 定义解释器身份，字面 link 只作 TOCTOU
证据；科学四格不变。runner/activation=`40e89c6a…ae09/e878de11…0447`，专项 `91 passed`，source gate
通过。Pod2 唯一 dry-run 报告四格且未创建 root；同一 SSH 中唯一 execute 自然结束、rc=`0`、无 retry
或 trainer/GPU/真机 signal，terminal summary=`ac880412…b7030c`。四格 generator/TOPP 均 rc=`0`，
正手×正手/反手 ready 为 `0.80/0.86 s`，反手×正手/反手 ready 为 `1.10/0.94 s`，因此 `<=0.5 s`
为 **0/4**。输入稳定性为空、`mjcf_closure_exact=true`，但 `physics_replay_exact=false`、
`source_closure_exact=false`、`strict_global_minimum_proven=false`；这只是 screening，不是完整动力学或行为
通过。当前 two-ready/hold/join family 按预注册规则停止且不重放；L0、vendor L1、桌网、动力学和行为卷
没有候选进入，仍全部开放，
所以 G08 不晋级。
见[实验](../experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md)和
[操作](../operations/run_ready_to_strike_motion.md)。

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
`configs/motion_video_gvhmr_s0_m0_results_20260713.json`。canonical-beta 后来已完成；本段当时尚无 GMR
结果，2026-07-20 回收的 exact-GMR completions 见下方更新。schema-2、simulator、RL 或真机仍无结果，
所以 G08 状态不变，也没有动作晋级。

这五条结果的下一层已收成两份 exact post-GVHMR handoff，并完成 runtime no-clobber consume；S0/M0
handoff exact SHA 分别为 `d57a93e0...a1054` / `60c55150...088ef`。其后的 canonical-beta 也已拆成两份
独立 prereg：只允许注入旧 exact donor，禁止重新聚合新五条；host 新旧专项 `15 passed, 1 skipped`，
真实一加四条 PT 已在绑定 Pod1 CPU runtime 完成 `inspect/consume`，completion manifest SHA 为
`964a7333...f1be3` / `5cef05f7...71a65`，non-beta 内容全 bit-exact。S0 的
`contact_truth`/效果继续为空且禁止借用拉球题；M0 的末态约束明确为去除公共 root、对齐 heading 后的
`right_foot_xy - left_foot_xy` 初始/终态稳健向量，横向站距与前后错位都要保留，脚并拢不算成功。
foot-site mapping 和数值容差已由下段 exact-GMR plan 预注册；本段当时尚无 runtime 结果，当前以
2026-07-20 completion 回收更新为准。详见
[post-GVHMR 卷宗](../experiments/motion_post_gvhmr_s0_m0_handoff_20260713.md)。
canonical-beta 的计划、运行边界与未来 A3 脚距 null contract 见
[canonical-beta 卷宗](../experiments/motion_canonical_beta_s0_m0_20260713.md)。

## 反手拉 B/C 主选整轨 SE(2) 实体化（2026-07-14）

B/C rank-0 主选不再停在“待写 materializer”：两份独立 no-clobber prereg 精确绑定 selection result
`8a80a409...8d2be`、各自 candidate/source SHA 和 counterfactual registry；同一 consumer 只把冻结的
XY translation + yaw 作为 proper [SE(2) 平面刚体变换](../DEFINITIONS.md)原子应用到整条 floating-root
轨迹。它不取 fallback、不改 Z/尺度/关节/逐帧/时间律，不运行 TOPP；xyzw orientation 按 yaw 左乘，
若 exact payload 带显式 world root velocity 则一致旋转。受限 pickle loader、未知字段 fail-closed、
save/reload 逆变换、刚体距离、保地、report-last/no-clobber 负测共 `10 passed`，全仓 host tests
`656 passed, 9 skipped`。

两份 exact 私有源先通过只读 inspect，随后在 Pod1 CPU-only runtime 完成 `consume`：B motion/report
SHA 为 `27827912...ad6` / `a238c077...df3`，C 为 `0dd981a6...f48b` / `b3b93d2c...f67`。这只发布冻结的
整轨 root 变换并解锁独立 schema-2 prereg；尚未运行 schema-2、L0、vendor L1、桌网整轨、动力学、
simulator/RL 或真机。因此证书仍为 0、G08 仍是 Research track；内部 materialization 失败必须停止
资产，只有后续外部桌/网失败才允许 selector 推进 frozen fallback。
完整命令与边界见[空间重定位实验](../experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)和
[操作文档](../operations/run_motion_spatial_retarget_screen.md)。

schema-2 前置源码审计随后闭环了一个会把 B/C 31 维关节列静默打乱的合同矛盾：GMR `dof_pos` 与
runtime/schema-2 `joint_pos` 不是同序。现在两份明确命名、内容绑定的表和双向 permutation 是唯一真源；
旧 YAML 与 `AGIBOT_A3_JOINT_NAMES` 只允许镜像 GMR source order。converter 直接消费新合同；历史 L0
auditor 因 executed prereg 的源码 SHA 绑定保持 byte-exact，其 runtime target literal 改由 validator AST
复核。完整 ONNX metadata gate 对重复/缺失/额外/错序/错误长度/partial metadata/非 finite 负测为
`12 passed`，基于 `origin/main@5734dc8` 的 repo-level 回归为 `733 passed, 10 skipped`。这没有消费
B/C 私有 PKL，也没有跑
FK/schema-2/L0/simulator/RL/真机，因此证书仍为 0；
下一步仍是独立 content-bound B/C schema-2 prereg，而不是直接训练。

该独立 prereg 的源码门现已完成：B/C 分别绑定 exact SE(2) PKL/report 与互不重叠的 no-clobber
输出，公共合同绑定 restricted pickle、formal donor SHA/三行 metadata 期望、vendor MJCF 的 `1 XML +
74 mesh` closure、31 关节映射、32 body order、30→50 Hz 和 link-origin pose/COM velocity。consumer
只接受 `--hope_frame off`，因此不会把已在 HOPE frame 的 root 再转一次。专项为 `17 passed`，两份
`static` 均未读取私有资产。donor metadata 仍只是绑定期望，不冒充 ONNX runtime receipt；下一门是
逐资产只读 `inspect`，之后才允许一次 schema-2/FK consume。当前没有 FK/NPZ、L0/L1、桌网、动力学、
simulator/RL 或真机证据，证书仍为 0、G08 状态不变。

两条只读 `inspect` 随后在 Pod1 的 exact detached `748b6d5` checkout 闭环。默认 Python 因缺
`onnxruntime` rc=2 fail closed 且未写；现成 `hope_mjeval_venv` 绑定 Python/NumPy/ONNX Runtime/MuJoCo
`3.12.3/2.5.0/1.27.0/3.10.0` 后，B/C 以 `91/98` 帧分别 rc=0，donor metadata 与 vendor model name
domain exact，output roots 前后均不存在，checkout 仍 clean。tracked receipt SHA 为
`8e2d2d2d...3fb61`。这只是 FK 的运行时前置：没有计算 151/163 帧 FK、没有 NPZ、没有 dynamics step。

下一层 v1 activation SHA `366d59d5...d6337` 因不能阻止 direct consume、失败清理后也不永久花掉
attempt，保持 **NO-CONSUME**。替代的 v2 activation/validator/runner 分别为
`72b22ccd...6ffb` / `3798122b...b536` / `8e66e050...a447`：atomic pre-child claim、B/C
shared flock、失败后不可恢复预算、completion-last success、current/historical runtime lineage 复核和
NPZ 内容级 schema-2/finite/body-order validator 已落源码。攻击负测含 bypass、并发、failure cleanup、
runtime/module-origin drift、缺失/伪造 NPZ，专项 `28 passed`、连同 prereg `45 passed`。runner 仍未在
Pod 执行，两个 output root absent、证书为 0；L0/L1、桌网、动力学、simulator/RL、正式动作和硬件仍
全部 false，G08 仍是 Research track。

其后 B 花掉唯一 v2 consume 并产出 exact runtime-order schema-2/FK 结果：151-frame NPZ、report、claim、
completion-last success 的 SHA 分别为 `e2eb99e6...d28cc`、`4f524593...bc38`、
`76e7ff88...fb0`、`c0a25f2c...f4f8b`，独立复核为 `runner_lineage=true`、`npz_bound=true`；C 仍保持
未消费后备。B 的下一张 [动作 L0 静态审计](../DEFINITIONS.md#motion-l0-static)已完成 source/static
门的 portability 修复：exact prereg SHA `7e155c89...97bc0`、validator SHA `5970f82b...b411`，两个
dependency-light 专项合跑 `51 passed`。首次 runtime 调用在任何运动学检查前暴露历史 runner 将 consume
checkout 绝对路径误当 body-order 内容身份；只创建 certificate 父目录，没有 certificate，也没有重跑。
本修复不改历史 runner/activation/claim 字节，按 claim 绑定的 activation bytes/SHA 与 source tuple保存
历史来源，再显式绑定当前 detached-clean commit/runner/source-validator/runtime-body-order；没有旧 Pod1
路径 fallback。原生 consume loader 仍拒绝当前 runner 接管旧 activation，C 没有消费授权。新 `dry-run`
执行完整只读 L0 但不写 certificate；attempt ID、receipt、claim/success、NPZ/report 仍严格绑定。
portable 修复后，V1 full dry-run 已在 Pod2 进入运动学门并以 float32 round-trip byte-equality
fail closed：position/quaternion/COM velocity/angular velocity 分别有 `537/917/1261/2320` 个 component
不逐字节相等，最大差 `1.1920929e-7 / 5.9604645e-8 / 2.9802322e-6 / 5.9679151e-6`；没有
certificate。源码证明 schema-2 只存 post-FK normalized float32 root body pose，V1 却把它当原始
free-joint qpos 再注入并要求 bit-idempotence；这不是动作、关节、接地或支撑脚失败。

V2 保留 V1 全部 exact lineage/MJCF/runtime、joint range、ground、support-foot、no-clobber 与 downstream
false gate，只把不可重构的 pose 比较改为 two-[`ULP`](../DEFINITIONS.md#float32-ulp) + physical cap，
COM velocity 改为 stored link pose + exact MJCF `body_ipos` 的 50 Hz 解析误差传播，angular/joint velocity
仍 byte exact。Pod2 exact detached `main@cc1a2b1` 的 full `dry-run` 已通过，随后唯一 no-clobber
formal audit 发布 certificate SHA-256
`60c08185e15c80621063bcedc65b42b6b738a12caeb8fb4e40a4c197e7daafc6`。**该 L0 子门已完成**；
证书只令 `vendor_l1_authorized=true`，桌网扫掠、动力学、simulator/RL、formal motion、Gate3 和真机仍
blocked。下一步是 vendor L1 整轨自碰/球拍自打，不得越级训练。边界、冻结输入与命令见
[L0 实验卷宗](../experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-L0.md)和
[L0 操作文档](../operations/run_motion_backhand_loop_b_l0_static.md)。

该下一门现已形成 [`vendor L1 safety audit`](../DEFINITIONS.md#motion-vendor-l1-safety) 的 source-only
预注册与 validator。它逐字绑定 L0 certificate `60c08185...afc6`、B NPZ `e2eb99e6...d28cc`、
vendor MJCF/75-file closure、compiled collision contract 和 exact CPU runtime；沿用已验证的
root linear / quaternion shortest-arc slerp / joint linear 插值，把 `151 @ 50 Hz` 全轨有限密扫为
`1201 @ 400 Hz`。所有 enabled robot geom 的穿透超过 `1e-6 m`，或球拍/拍柄距 head/trunk/对侧臂/
右肩三轴与右肘/下肢小于 `5 mm`，都会否决整条 B，不能由 warning/reward/其他分数补偿；`5–20 mm`
只登记 warning。hard 判定直接调用 exact/saturating `distance >= 5 mm` predicate，不用 `1e-4 m`
bisection midpoint；4.99/5.00/5.01 mm 边界反例已闭环。右腕/手/球拍安装链只从 `5 mm` proximity
pairs 排除，enabled-robot 实际穿透仍 hard fail。dry-run 也必须在 runtime 前证明输出 parent 是已存在
真实目录且 target absent/非 symlink。该有限扫掠
**不是**数学连续时间证书，且不含桌网或动力学。两次 Pod2 runtime 假拒绝修复后，exact
`main@7dec698` 的 full `dry-run` 与唯一 no-clobber audit 已通过：1201 样本自碰/`<5 mm` 自打/warning
=`0/0/0`，最小余隙 `0.1382918358 m`，certificate SHA-256 `6840df34...db60`。B 现只解锁独立桌网
整轨门，动力学或训练仍不得越级；G08 仍是 `Research track`，该动作子项未闭合。见
[L1 卷宗](../experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1.md)和
[L1 操作](../operations/run_motion_backhand_loop_b_vendor_l1_safety.md)。

首次 Pod2 CPU `dry-run` 随后在真正轨迹审计前抓到 harness import blocker：冻结 grounding helper
被要求用只供本审计隔离的 private module name 经 `sys.path` 导入，因此 `ModuleNotFoundError` fail
closed 且没有 certificate。修复改为按 plan 冻结 bytes/SHA 从 exact path 事务式加载；加载前后复核
内容和 module origin，失败恢复原 `sys.modules` entry，防止 stale/半初始化 module 污染。真实 helper
private-alias 回归及 SHA/body-failure 反例已过，source/static gate 重验通过；合入后的 runtime 重跑已越过
该 import 点，并暴露下一段所述 joint-order adapter blocker。B 始终停在同一门。

import 修复进入 `main@b75204d` 后的第二次 Pod2 CPU `dry-run` 又在 range gate 假拒绝：schema-2 NPZ
是 runtime/Isaac joint order，L1 却把 31 列直接交给只接受 vendor MJCF/GMR hinge order 的 grounding
helper。dense 704 正是 source frame 88；runtime column 23 的 elbow `1.1804603338 rad` 被误标为 ankle，
相对 ankle 上限 `0.523599` 恰好得到所报 `0.656861334 rad`，真正 column 14 ankle
`-0.5744639635 rad` 合法。L0 按 name→MJCF id 映射，故 certificate 的 `max_excess_rad=0.0` 与原动作
并不矛盾。L1 已改为在 densify/range/qpos 前执行冻结名字表的 byte-preserving 31-joint 双射并报告
permutation，duplicate/missing/extra/order-contract drift 均 fail closed；没有放宽 range、修改 B 或启用
C。修复后 runtime/certificate 结果见本节上方；原 dense704 只作为假拒绝根因证据。

B 的下一道整轨桌网余隙门现已完成 source/static 预注册，并做过一次未进入轨迹循环的 exact MuJoCo
`dry-run`；该次只暴露 harness 假拒绝，尚无动作 runtime 结论。它逐字绑定
现存 vendor L1 certificate `6840df34...db60`、B NPZ、L1 plan/validator、vendor MJCF/75-file closure，
并在消费前要求 certificate 的 `table_net_authorized=true`。坐标只允许 canonical HOPE 桌坐标到
schema-2/MJCF 世界的无旋转纯平移 `(+0.5,+0.7625,+0.76)`；因此桌板、网和两根网柱在同一冻结 tracking
桌位中检查，而不是从空挥视频背景猜 capture extrinsic。validator 把四个 world-fixed box 追加到 canonical
worldbody，在内存中用 exact 74-file asset map 编译；每个 `1201 @ 400 Hz` 有限样本检查 `37×4=148` 个 pair，球拍/拍柄另做
汇总但不从全机器人门排除。`<5 mm` 是无 epsilon 的 finite threshold hard fail，不能由其他帧/reward 补偿；
有限密扫仍不是数学连续时间证明，且没有动力学、训练或真机含义。首次 source 版本被红队证明存在
certificate/NPZ/MJCF 的 check→reopen 和输出 parent path-swap 窗口，因此明确 NO-MERGE；修复版把每个
runtime 输入绑定为 `O_NOFOLLOW` fd 的单次 bytes snapshot，canonical XML + 74 mesh 从 pinned model-root
dirfd 读取，输出用 pinned parent dirfd + `openat(O_EXCL)` + file/directory `fsync` 后复核同一 inode/bytes。
certificate swap、model-root replacement、output-parent swap、duplicate ZIP、错误 dtype、NaN/Inf 均有
fail-closed 负测。第二次红队又证明完整 phase/self-collision module exec 会消费未绑定普通 project import；
修复后只保留与 exact upstream AST 等价的四个 dependency-free phase kernel，以及有 upstream 数值 parity
的本地 MuJoCo saturation/bisection kernel，不再执行 `virtual_return_scorer`/`audit_motion_npz` 等 transitive
module；冻结 source validator 调用后还原完整 `sys.path`/`sys.modules`，legacy L0 runtime module 也不再
执行。伪造 `sys.modules` 注入已 fail closed；二分 midpoint 与 certified lower bracket 分字段，网柱尺寸和
左右 placement 也新增 exact `table_tennis_env_cfg.py::build_net_post_cfg` source binding。旧版
dependency-light 专项 `29 passed`、source/static 命令 PASS；首次 Pod2 `dry-run` 随后证明旧 harness 把
global geom ID 当成 robot identity：MuJoCo 先编号 worldbody 自身的 floor+四个新增障碍，所以所有
child-body robot geom 确定性整体 `+4`，并非 robot 几何漂移。schema-v2 只接受 floor=`0`、
obstacle=`1..4` 和 robot 精确 `+4`；同时逐项守住 robot 相对顺序/名字、root/joint topology、`qpos0`、
37 个 collision row/mesh 与 canonical collision SHA，任意其他 permutation 或 `1e-9` collision drift
仍拒绝。v2 focused `36 passed`、完整 B lineage chain `137 passed`、source/static PASS。v2 合入
`main@f214a80` 后的 Pod2 `dry-run` 又在几何循环前 rc2 fail closed：exact 724-byte runtime joint-order
文件含一行 `#` 人话头和 31 个唯一关节名；upstream L0 parser 明确忽略 blank/comment，table/net snapshot
reader 却只忽略 blank，误把注释算成第 32 个关节。90-byte log SHA-256 `5c9a5940...f92d`，certificate
absent；这不是 NPZ/order/bijection 漂移。schema-v3 逐字复制 upstream comment 过滤语义，仍拒绝额外未标注
metadata 与 duplicate name；production-shaped focused `39 passed`、完整 B lineage chain `140 passed`、
source/static PASS。v3 随后在 clean `main@b9b011b` 发布了 hard/warning=`0/0` 的旧 certificate
`39d6cc38...79a19`，但独立 exact-semantics review 将其拒绝：reporting-cap predicate 只证明
`distance>=0.1-1e-12`，旧 aggregator 却在所有 pair saturated 时把默认 `0.1` 冒充 certified lower
bound。hard/warning 结论不变，但旧 `table_net_complete/dynamics_authorized` 不接受。schema-v4 将 pair/
全轨 saturated 初值改为 `0.099999999999`，保持 midpoint/pair/time=null 并显式标 saturation；旧 cert
path/bytes/SHA machine-bound 为 rejected，新输出使用独立 `table_net_primary_v4` namespace/name。
初版 v4 commit `7241157` 又在合入/运行前被红队判 NO-MERGE：hard/warning 仍错误复用 reporting
`threshold-1e-12` predicate，可接受 5 mm/20 mm 各自门槛下方半个 epsilon。当前 v4 把安全门改为
finite `distance>=threshold`、non-finite fail closed，只让 epsilon 留在 reporting cap/bisection；5 mm 与
20 mm 的 nextafter-below/half-epsilon-below 反例均闭环。focused `47 passed`、完整 B chain `148 passed`、
source/static PASS。随后 clean Pod2 `main@c047ea7` 的 full dry-run 与唯一 v4 audit 均通过：`1201 @ 400 Hz`
逐帧 `37×4` 对，hard/warning/unsafe=`0/0/0`；certificate `93fd5435...9b0e7` 对 saturated reporting 只声明
`0.099999999999 m` 下界且 pair/midpoint/time=null，已由独立复核接受。B 因此完成桌网有限密扫并只解锁
下一道 vendor 动力学/平衡门；连续时间、动力学行为、RL、Gate3 与真机仍未证明，所以 G08 仍是
`Research track`，该动作子项未闭合。
见[桌网实验卷宗](../experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET.md)和
[操作文档](../operations/run_motion_backhand_loop_b_table_net_clearance.md)。

S0/M0 的下一层 exact GMR 现已冻结成两批独立 machine plan。共享 consumer 不把 retarget body order 猜成
canonical vendor body order，而是分开绑定两套 XML order，再用显式 31-joint bijection 接入 A3 FK。M0
四条候选的人工 ready windows 已展开为 exact 30 Hz sample indices；结果将同时报告初末前后脚错位和横向
分离，并用独立 5 mm narrowing 门阻止“落在 3 cm ready band 内但实际收脚”的假通过。S0 仍保持
contact/effect 为 null、禁止借用拉球题。

2026-07-14 第二轮低频只读补证已闭合原 16 项 exact runtime 清单：七个 import module、mapping、Python/pip
origin、direct retarget 31-joint/32-body preorder 与 31-index bijection 都已绑定。direct retarget XML 的 site
inventory 精确为空，`left_foot/right_foot` 都 absent；consumer 已改为拒绝 canonical vendor sites 冒充
retarget sites，M0 stance 仍只在绑定的 canonical vendor MJCF 上做 FK。两份 batch plan 与 shared runtime
均为 `preregistered_not_executed`，两次 host `static` 均 `PASS`，专项 `13 passed`、仓内 `tests/` 回归
`867 passed, 10 skipped`。

真实 v1 S0 runtime inspect 随后在创建 root 前抓到一个证据合同错误：只登记的 pip-freeze SHA
`97c66009...18ff` 无对应 line list/bytes，无法由 exact Python 的实际 234 行规范化快照
`56b0f8af...c694` 复现。M0 没有重复 shared blocker；两份 v1 root 均 absent，v1 永久
**NO-CONSUME**，不能把这个负结果写成动作或横移失败。

新的 attempt v2 使用新 consumer/plan/runtime 和 `exact_gmr_v2` root。它把 4,702 bytes snapshot 本身纳入
source，并绑定冻结 v1 base consumer 和五个直接 import 的 version、origin、`METADATA/RECORD`；RECORD
origin、duplicate JSON、S0/M0 shared exclusive flock 与 post-converter 重验都 fail closed。两份 host
`static-v2` 已通过，v2 专项 `15 passed`、新旧 focused `28 passed`、仓内回归
`949 passed, 10 skipped`；没有 runtime consume、脚距结果或动作晋级，因此 G08 状态不变。

这只解除 v2 source/static blocker；除上面的 v1 no-write inspect 与只读 v2 closure 取证外，没有运行
v2 runtime `inspect/consume`、GMR、schema-2、安全、动力学、RL 或真机。见
[exact GMR 卷宗](../experiments/motion_exact_gmr_s0_m0_20260713.md)。G08 状态不变。

2026-07-15 的 Pod2 v2 runtime 尝试进一步把 blocker 缩到解释器存在性：clean detached `b75204d` 上
S0/M0 两份 `static-v2` 再次 PASS，plan SHA exact，但合同写死的
`/workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10` 与父环境均不存在，故 rc127 发生在
consumer 前。恢复审计又确认 Pod2 缺 exact GMR tree/283 MB bundle、SMPLX/model/mapping 和七份 S0/M0
canonical 输入；现有 Isaac venv 只与冻结 234 行 snapshot 精确重合 87 行。两个 v2 root 与 shared lock
前后均 absent；没有 GMR 或脚距结果，也不能偷换另一 Python。下一步是权威资产恢复后新建隔离、可重建
runtime v3，不是重跑 v2 或放宽动作门。这是当时基于 Pod2 单点证据的恢复设想；2026-07-20 回收 Pod1
completion 后，当前下一步改由下节的动作/stance 修复决定。G08 仍是 `Research track`。

### 2026-07-20 S0/M0 exact-GMR completion 回收

后来从 Pod1 回收的 report-last manifests 纠正了“两个 v2 root 全局 absent / 未 consume”的推断；上段
Pod2 rc127 仍是另一处真实失败 location。S0/M0 都已达到 `complete_exact_gmr_diagnostic`：S0 的
88-frame 输出通过 finite/30 Hz/31 DoF 结构门，但 ball contact/effectiveness 仍为 null，必须另建高球拍压
题族；M0 四份 moving 输出通过同一结构门，却全部失败 frozen stance gate (`0/4`)，因此 input-gate
rejected/no-RL。下一步是保留横移同时修复末态回到各自动作初始 stance，再重走 exact GMR；两批的
formal/schema2/training/hardware 都仍为 false，G08 状态不变。详细 manifest/output SHA 与逐条失败值只在
[exact GMR 卷宗](../experiments/motion_exact_gmr_s0_m0_20260713.md)维护。

## 文档路由更新（2026-07-12）

G08 仍是长期 blind-spot 路线图。当前已采用 setting、阶段/小目标构成和 feature 决定统一放在
[`docs/NOW.md`](../NOW.md)；动作/动作库/恢复实验的详细记录放在
[`docs/experiments/`](../experiments/README.md)。本次文档修改没有运行模拟、训练或真机测试，
也不改变 G08 的 gate 状态。
