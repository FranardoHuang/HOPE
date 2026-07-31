# 标准工序：发一波消融

**每步 = 做什么 → 通过条件。** 不过就停，不许跳步补位。
排序/算力/seed 纪律看 [runbook](../runbook.md#统一队列排序与算力纪律)，本页只管发射动作。
本页把散在 `run_phase1_push_robustness_wave.md`、`run_phase1_balance_temporal_matrix.md`、
`run_on_runpod.md` 和 2026-07-26 实验记录里的同一套纪律收成一处。

## 发射前

1. **认领**：从 [NOW 唯一队列](../NOW.md#统一工作队列唯一优先级账本)取最靠前且依赖已满足的项。
   不许建第二个队列。
2. **冻 commit**：clean detached exact commit，写进每臂的 `source_commit.txt`。
3. **渲染命令到文件**：队列渲染器把完整 argv 写进 `arms/<job>/argv.txt`。
   **⚠ argv 模板必须取自引号完好的命令文件，不得取自进程表**——从 `ps` 抄会丢 shell 引号，
   Hydra mapping 被打碎（2026-07-26 首发即死，同波内又犯一次）。
4. **启动锁存在**：`ssh <pod> 'test -x /workspace/bin/kit_boot_lock.sh && echo LOCK_OK'`。
   **不用** `launch_kit_training_locked.sh`（其 `180 s` stale 门是 Wave A v8/v9 死因）。
5. **run 目录 no-clobber**：所有 run_dir 与 probe dir **都不得已存在**；创建用一次原子 `mkdir`。
   基础设施失败（stale/SIGABRT/身份竞态，**非** NaN/OOM 科学失败）允许在 fresh namespace 下
   逐字 retry 一次（`_r2` 后缀），同 phase 第二次失败转根因线；原 namespace 永久只读。
6. **GPU 真的有位**：`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`，
   目标卡 **compute PID < 4**（4 条/卡是上限）。
   - 同时看利用率和日志活性，**不只看显存或 PID**；重复行按唯一纯数字 PID 计数。
   - 卡上现有进程**先 ps 认领**：谁的、活的还是死的（僵尸评估占槽让发射器死等 3 小时）。
   - **绝不抢占在跑的格**。快照会过期，每次发射前重查。
   - ⚠ 旧的"每卡零 compute 进程"口径已被推翻——它导致每卡第一格上卡后其余三格被拒。

### 未变配方的诊断续跑快线

下面六项逐字不变时，fresh diagnostic/canary 续跑不重复做 repin、动作 bundle 物化或 host
大联合回归：

1. exact source commit；
2. launcher 路径与 bytes SHA；
3. ordered action bundle 路径与 bytes SHA；
4. effective Reward recipe SHA；
5. PPO/policy contract SHA；
6. diagnostic/formal 与 fixed/dynamic curriculum 模式。

续跑只做四件事：回读 GPU owner/UUID、创建 fresh no-clobber namespace、渲染并保存 canonical
plan/argv、用 `/workspace/bin/kit_boot_lock.sh` 串行启动。任一身份字段改变，只重做受影响的
物化和专项检查，不把无关 host 套件重新跑一遍。

这条快线只购买“能否优化并产出 policy”的诊断证据；它不能把
`training_authorized=false`、关闭 formal evidence fence 或冻结课程的运行写成正式 Gate/curriculum
晋级证据。NaN/Inf、真实 hard limit、table hit、fall 和动作/Reward/PPO 身份漂移仍是硬停止条件，
不得借快线关闭。

小时巡检不能只看一次 NVML 利用率或 `run.log` 修改时间。对每个绑定 namespace，必须从所有新增的
完整 RSL-RL iteration block 记录：

- update、collection、learning 墙钟；
- [`collection_vector_step_wall_s`](../DEFINITIONS.md#collection-vector-step-wall-s)：
  `collection_wall_s / num_steps_per_env`；
- [`amortized_e2e_vector_step_wall_s`](../DEFINITIONS.md#amortized-e2e-vector-step-wall-s)：
  `iteration_wall_s / num_steps_per_env`；
- [`collection_environment_step_us`](../DEFINITIONS.md#collection-environment-step-us)：
  `collection_wall_s × 10^6 / (num_envs × num_steps_per_env)`；
- [`collection_environment_steps_per_s`](../DEFINITIONS.md#collection-environment-steps-per-s)：
  `(num_envs × num_steps_per_env) / collection_wall_s`；
- terminal、qdes/actual hard-limit、reference、table/fall 和 strike opportunity。

`num_steps_per_env` 必须从该 run 的 contract/agent receipt 读取，不能在巡检脚本里静默假设 24。
reason mask 可重叠，禁止相加冒充 terminal。已绑定 namespace 没有匹配活进程时必须报
`MISSING/EXITED`，不能因为 `ps` 没列出就从报告中消失。当前任务只允许显式 Pod1 target；
no-argument 巡检不得自动读取过期 wave 文件或连接 Pod2。

同卡第二条 operator-direct 诊断只用于 breadth，不是 canonical/formal 发射。它仍须保存 exact
plan/argv/identity、fresh no-clobber namespace，并用 Kit boot lock 串行启动；一旦在 scene
construction 或 PhysX start 停止前进，保留日志后按 exact PGID 关闭，不循环重试。`4ff48b21`
的 task-strong direct 就在 PhysX start 活锁，故未算作活跃 Reward 臂。

### 智元 A3 vendor N1 单卡诊断

[智元基线 N1 单卡诊断](../DEFINITIONS.md#n1-vendor-baseline-diagnostic)是本页“每卡最多四进程”
宽度规则的显式例外：它要求目标物理 GPU 为空，一条 run 独占一卡和
`/tmp/hope_lean_queue_gpu<N>.lock` 整个生命期。它不改 formal trainer+evaluator 的双 GPU 合同。

当前 fresh-training authority 是新
[`HOPEPingPongActionBallA3VendorV1`](../DEFINITIONS.md#a3-vendor-v1-profile) plant 和
2026-07-31 尽调；绑定仓库旧常数的审计只是历史证据。一次性
[`A3 vendor identity smoke`](../DEFINITIONS.md#a3-vendor-identity-smoke) 已在 exact source
`5665963e96bf75c677e7669efc58c449e0c04876` 完成 recipe-only 和 `1 env×2`：
schema-3 training contract SHA 为
`98fa3239daba825f07d3997fb28f4564c92967536f2552e6bdc0f8772781366f`，
`model_0.pt`/`model_1.pt` finite，delay/ABI/std marker 计数 `1/1/2`。authority
live-order bug 已修复。shared-ready policy recipe SHA
`27bf405e5677fe2e7bab6fcc15c166901734048dd334b8b0abc3a8ffef3ce416`
不是 dynamic-ready recipe，不得复用。

现已物化的 `bh_loop_c` 证据集是：

- dynamic-ready candidate SHA
  `c831a4e6d1c03519181efb090120a881702d113e95ebcf22f745a3a2ca4fc794`;
- nominal-hold receipt SHA
  `11c025dc25cba93c7d0d9894bac75da05a1a7aff11f797e9a35f9b2906f67740`，
  `0.8 s / 40` 步 PASS，feet-contact `1`，无 terminal；
- bundle SHA `9881c52ca035bbdee0a3e1d0c0689eb7592b2a73b5442866a9a6e9480cbaae03`；
- actual-authority receipt SHA
  `f66a9e59f441c22c465d3236d717c95354393d04c5975f58ece3e7612a65461a`;
- materialized required-identity SHA
  `240f3757e45006de9dc5f4ecabcfc40071058009751fd1f0b8eb92656e1801ff`，绑定
  contract `98fa3239…` 且只允许 `bh_loop_c` dynamic-ready action。

本批 launcher 同时 pin required-identity 和 actual-authority SHA，且相关
materialization/pin 改动已过 `90` 个非 Torch 定向测试并随本批跟踪。clean
`e7787e25` 已物化 dynamic-ready policy `e408b845…c65d`，随后的
`bh_loop_c` diagnostic `smoke` claim=`be783ab7…ad54` 完成 `1 env×2`，两份
checkpoint finite，ABI/delay/std-LR marker=`1/1/2`。same-seed `4096×5` probe 也已自然
完成，但 `14,086` 次 actual-hard terminal 使它明确失败，不能物化 long gate receipt。
下一 source 身份强制 stable-ready、保留 `std=0.075 m` 精核并叠加
`std=0.30 m` 粗核，同时把 plant-state guard 从 2% 提前到 5%。必须重物化后
按 `recipe-only → 1×2 smoke → 4096×5 probe → 4096×32 push_evidence`
串行；不得跳到 `long`；formal、promotion、export、judge、deployment 和
hardware 均未授权。

首个 r1（source `2430fbb2`、claim `e37f8169…e32`）已永久 spent：它在
schema-v2 pre-scene 验证之后，因 MotionCommand consumer 只接受 schema-v1 而
fail-closed，零 recipe/零 PPO。不得编辑/重用 r1 spec 或 namespace；必须等兼容
v1 且完整验证 v2 plant/delay 的 consumer 修复进入 clean commit，再生成 fresh spec/claim。

下面的 bootstrap repin 和 identity-smoke 命令保留为精确可复现记录，不是重跑授权。

profile pins 与 repin producer 必须先在同一 clean commit 中。从该 commit 只运行一次：

```bash
SRC_COMMIT=$(git rev-parse HEAD)
git diff --quiet
git diff --cached --quiet
python3 hope_training/whole_body_tracking/scripts/materialize_a3_vendor_identity_manifest.py \
  --repo-root "$PWD" \
  --source-commit "$SRC_COMMIT" \
  --source-manifest configs/n1_contact_20260730_stable_v2/bh_loop_c.manifest.v3.775f74183e58.json \
  --expected-source-manifest-sha256 775f74183e58683df48f5f44084e89320736d1533a4d962f43f455664830d8e5 \
  --profile-pins configs/a3_vendor_identity_bootstrap_20260731/action_ball_profile_pins.v1.07e79f968a63.json \
  --expected-profile-pins-sha256 07e79f968a6301f17a932775586868aa96be8c2df3bcf0358cab096280857f10 \
  --prototype-output configs/a3_vendor_identity_bootstrap_20260731/bh_loop_c.vendor_identity.prototype.v2.json \
  --manifest-output configs/a3_vendor_identity_bootstrap_20260731/bh_loop_c.vendor_identity.manifest.v3.json \
  --receipt-output configs/a3_vendor_identity_bootstrap_20260731/bh_loop_c.identity_bootstrap_repin.v1.json
```

三个输出必须在后续 artifact commit 中同时跟踪；任意目标已存在时整次拒绝。
producer 会在写入前用 exact source commit 重跑正式 pinner 并要求 profile bytes 逐字相等，
所以人工重签 payload 不能过门。

identity-smoke 只从 clean exact commit 生成 canonical spec。以 Pod1 GPU0 为例，先生成
recipe spec（`template` 会自动钉住 Reward SHA，不接受任意覆盖）：

```bash
python3 hope_training/whole_body_tracking/scripts/launch_a3_vendor_identity_smoke.py template \
  --stage recipe \
  --checkout /workspace/franco/a3vendor_<short-commit> \
  --commit-sha <full-40-hex-commit> \
  --isaac-python /workspace/hope_isaac_venv/bin/python \
  --gpu-index 0 \
  --gpu-uuid GPU-889b1712-8d89-0536-5c9e-e79aae30523d \
  --owner Franco \
  --namespace /workspace/franco/a3vendor-identity-recipe-<short-commit>-gpu0-r1 \
  > /workspace/franco/a3vendor-identity-recipe-<short-commit>-gpu0-r1.spec.json

python3 hope_training/whole_body_tracking/scripts/launch_a3_vendor_identity_smoke.py \
  plan --spec /workspace/franco/a3vendor-identity-recipe-<short-commit>-gpu0-r1.spec.json

python3 hope_training/whole_body_tracking/scripts/launch_a3_vendor_identity_smoke.py \
  launch --spec /workspace/franco/a3vendor-identity-recipe-<short-commit>-gpu0-r1.spec.json \
  --confirm-claim <plan-printed-launch-claim-sha256>
```

recipe 自然退出后，从 namespace 中 fresh
`vendor_shared_ready_policy_recipe.json` 取 `policy_contract_sha256`，再用同一
`template` 命令改为 `--stage smoke --policy-contract-sha256 <sha256>` 和 fresh
`a3vendor-identity-smoke-*` namespace，然后重复 plan/launch。两阶段都不得复用
namespace；任一阶段没有自然退出、finite 产物或 exact runtime receipt 时就停在该门。

已验证 revision 的 canonical spec 仍只允许 `bh_loop_c`，`bh_block` 机械拒绝；seed 只能选
`0/1/2`，`reward_profile=vendor_task_defaults`。2026-08-01 的 source-only successor 已把
`bh_block` 接入动作专属 registry、identity、authority、dynamic-ready 与 gate 链，但缺任一
新物化工件时必须 fail-closed，且尚无 Pod 证据、尚未采用；不得把“源码支持”误写成当前 revision
已授权。当前必须先过 dynamic-ready recipe-only
门，再串行开 `smoke`；`smoke` 证据合格前不得进 `probe`。第三个只记录未来目标：

- `smoke`：`1 env × 2 update × save1`；
- `probe`：`4096 env × 5 update × save1`；
- `push_evidence`：`4096 env × 32 update × save8`，只用于证明 exact vendor push
  真正执行过，不是额外学习比较；
- [`long`](../DEFINITIONS.md#n1-diagnostic-long)：目标预算为 `4096 env × 20001 update × save100`，
  但当前 launcher revision 机械拒绝；后续须同 policy/source/seed 的健康 probe
  与 push-evidence 共同产出具名 receipt。

没有任意 Hydra override，argv 必须直接继承 task leaf 的 startup Kp/Kd、
[`axis_box_6d_v2`](../DEFINITIONS.md#axis-box-6d-v2) 与
[`[0,2]` 控制步延迟](../DEFINITIONS.md#control-step-action-delay)，且必须保留
`diagnostic_unauthorized=true`。运行命令只有：

```bash
python hope_training/whole_body_tracking/scripts/launch_n1_vendor_baseline_diagnostic.py \
  plan --spec <absolute-canonical-spec.json>

python hope_training/whole_body_tracking/scripts/launch_n1_vendor_baseline_diagnostic.py \
  launch --spec <absolute-canonical-spec.json> --confirm-claim <plan-printed-claim-sha256>
```

Pod1 只允许串行 Kit boot：GPU0 seed0 看到真实 `Learning iteration` 与 exact identity 后，
才启 GPU1 seed1，再启 GPU2 seed2。三卡都先 smoke→probe。probe 的
[入窗拍距](../DEFINITIONS.md#strike-window-entry-distance)若多数 `>0.20 m`，立即转粗+细核修复，
不启动 long。stage-evidence v4 已消费 delay stdout receipt，focused `51 passed`；smoke/probe
仍仅用于机械诊断，`long` 还必须持有实际 probe 后生成的命名
`vendor_probe_gate_receipt`，否则
launcher 机械拒绝。这三条永久是
diagnostic-only，不得写成 formal N1、curriculum promotion、
export 或 judge 证据。

同一 source-only successor 还定义 `bh_loop_c` 的 fresh-only 单调自适应 σ canary：位置、速度、
拍面法向从 `0.20 m / 1.0 m/s / 0.52 rad` 只收紧到
`0.075 m / 0.5 m/s / 0.262 rad`，位置/速度/法向的成对 term 必须锁步，static 路径仍使用
`0.30 m` 粗位置核。该 canary 禁止 resume；发 spec 前必须从 clean exact source 运行零 PPO 的
hash-only Reward 物化，拿到 effective Reward SHA 后重钉 identity、authority、dynamic-ready、
bundle 和 launch claim。该阶段若缺 marker/SHA 或检测到脏工作树就停止；不能用手填 SHA，也不能
把 host `322 passed` 当作 Pod smoke。shared actual-hard 门未闭合前，static 与 canary 都不得
进入 long。

#### corrected-nominal `_r2` 物化与三 lane 模板

本轮不在旧 `a3_vendor_runtime_contract_20260731` 路径原地覆盖。可测工具 source
提交后，registry 必须单独进入 `_r2` fixed paths 且待产物 SHA 为 `None` 的
fail-closed commit。这个 commit 可用于物化，不可用于训练。先用各动作 identity
recipe/smoke 产生真实 live schema-3 contract，再从同一 clean checkout 运行：

`_r2` formal profile pins 由 clean `9a7429f1…` 的 seven-source blob map 产生，文件
SHA-256 为 `df7fe0f038d79e3a89feebc638eea48290caa7e8cf85c4ddefe76ac310b9d3fe`；
launcher 只认跟踪的
`configs/a3_vendor_identity_bootstrap_20260731_r2/action_ball_profile_pins.v1.json`。

```bash
LOOP_SOURCE_ROOT=/workspace/franco/a3vendor_<short-commit>_loop
BLOCK_SOURCE_ROOT=/workspace/franco/a3vendor_<short-commit>_block
SOURCE_COMMIT=<full-40-hex-commit>
ISAAC_PY=/workspace/hope_isaac_venv/bin/python

# 两个动作必须是同一 SOURCE_COMMIT 的两个 detached clean worktree。
# 第一次 materialize 会使其 worktree 产生 Git-visible 文件；若两次共用一个
# checkout，第二次必然按 dirty-source 拒绝。
for SOURCE_ROOT in "$LOOP_SOURCE_ROOT" "$BLOCK_SOURCE_ROOT"; do
  mkdir -p \
    "$SOURCE_ROOT/configs/a3_vendor_runtime_contract_20260731_r2" \
    "$SOURCE_ROOT/configs/a3_vendor_runtime_authority_20260731_r2"
done

"$ISAAC_PY" \
  "$LOOP_SOURCE_ROOT/hope_training/whole_body_tracking/scripts/materialize_a3_vendor_required_identity.py" \
  --repo-root "$LOOP_SOURCE_ROOT" \
  --source-commit "$SOURCE_COMMIT" \
  --action-id bh_loop_c \
  --live-training-contract /workspace/franco/evidence/bh_loop_c.live.training_contract.json

"$ISAAC_PY" \
  "$BLOCK_SOURCE_ROOT/hope_training/whole_body_tracking/scripts/materialize_a3_vendor_required_identity.py" \
  --repo-root "$BLOCK_SOURCE_ROOT" \
  --source-commit "$SOURCE_COMMIT" \
  --action-id bh_block \
  --live-training-contract /workspace/franco/evidence/bh_block.live.training_contract.json
```

脚本只会安装到 registry 声明的 fixed paths；不接受 `--output` 或 expected SHA。
任一目标已存在、contract/action/source/joint order 不对、或双输出中任一写入
失败，整次拒绝或回滚；不得手工搬 JSON 冒充产物。
identity launcher 的 `launch accepted=true` 只证明 Kit boot marker 已出现；必须按
`run.log.launch.leader.json` 的 exact PID/PGID/starttime 等待整组自然退出，再读
recipe/contract/checkpoint。禁止在前一组仍存活时开下一 stage。

在回填 runtime-contract/required-identity/authority/bundle/policy/Reward SHA 的后续 clean
artifact/source commit 上，三条人话 lane 与 code id 固定为：

| 用途 | `--lane` | action | sigma |
| --- | --- | --- | --- |
| 反手拉静态主臂 | `bh_loop_c_static_v1` | `bh_loop_c` | static |
| 反手挡静态主臂 | `bh_block_static_v1` | `bh_block` | static |
| 反手拉单调 sigma canary | `bh_loop_c_monotonic_fresh_canary_v1` | `bh_loop_c` | `0.20/1.0/0.52 → 0.075/0.5/0.262` |

smoke/probe/push 直接生成完整 canonical spec；以 loop static probe 为例：

```bash
"$ISAAC_PY" \
  "$SOURCE_ROOT/hope_training/whole_body_tracking/scripts/launch_n1_vendor_baseline_diagnostic.py" \
  template \
  --lane bh_loop_c_static_v1 \
  --stage probe \
  --output /workspace/franco/specs/bh_loop_c_static.probe.json \
  --checkout "$SOURCE_ROOT" \
  --commit-sha "$SOURCE_COMMIT" \
  --isaac-python "$ISAAC_PY" \
  --gpu-index 0 \
  --gpu-uuid GPU-449c8b80-f4a6-2d03-6e8a-b8ac68dea23d \
  --owner Franco \
  --namespace /workspace/franco/runs/bh_loop_c_static_probe_<short-commit>
```

long 必须先用 `--scientific-only` 产生不含 operational placement 的 tracked skeleton，
提交后再以 `--scientific-template <tracked-file>` 和 source/GPU/namespace 生成仓外 runtime
spec。不得直接跟踪含 `source.commit_sha` 的 full spec，否则会形成 Git 自引用。

三 lane 共用的 2% Hctrl 机械门在任何 `4096×5` 前先跑：

```bash
"$ISAAC_PY" \
  "$SOURCE_ROOT/hope_training/whole_body_tracking/scripts/probe_a3_vendor_dual_position_envelope.py" \
  --source-root "$SOURCE_ROOT" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --motion-file "$SOURCE_ROOT/assets/motions/fivebind_20260727/bh_loop_c_upper_stable_v2.npz" \
  --device cuda:0 \
  --output /workspace/franco/evidence/a3_dual_envelope_stress.${SOURCE_COMMIT}.json \
  --execute \
  --confirm SIM_ONLY_A3_DUAL_POSITION_ENVELOPE_8ENV_ONE_TICK
```

输出必须在 source 与 Isaac Lab 两棵树外且事先不存在。PASS 要求四个
joint/side pair 均是 ON 被 Hctrl 捕获、OFF 进入 `[Hctrl,Hmech)`、Hmech 严格不触边，
并且 finally 后全 env Hctrl exact readback 恢复。失败 receipt 只是诊断证据，不得
通过改 tolerance、actual-hard 定义或加 acceleration/jerk governor 伪造 PASS。

2026-07-31 首轮清场只处理有 exact sidecar 的旧残留：Pod1 GPU2 已按 sidecar `TERM` 并确认释放，
GPU0/GPU2 当前可用；GPU1 缺 sidecar，保持未动。缺所有权证据的进程不得为了凑三卡而终止。

### ActionBall A3 upper q/qd 修复与 hot-path 快线

`eaf55fba` 已把 recoverable 2%-inner occupancy 从 Done 中拆出，但 Pod1 4096-env 前五轮仍有
`2.5k--4.2k/update` 的 raw-hard terminal，且 q_des projection/nonfinite 为零、episode 未到
`t_hit`。当前禁止继续靠放宽 actual hard edge、加 env 或改 Reward 掩盖该反例。旧
`canonical_ready_v1` 的零脚接触只描述 donor 本身，不能外推现役 upper。qvel-only successor
虽修正了恒定腿 qpos 与 stale qvel 的 schema 矛盾，但后续双动作 4096×5 已证明它不是 reset
根因：loop/block episode 约 `23/12` 步、strike 恒零，actual raw-hard 仍爆炸。真正合同缺陷是
该几何双脚接触的深蹲前倾 pose 不是 exact A3 runtime implicit-PD 的闭环 stable hold。
`candidate_id=G1` 只是 A3 候选代号，不是机器人型号。

发射顺序：

1. qvel-only 是已完成但被行为 probe 否决的 schema 清理，不再作为 long 输入。upper successor
   必须保留 head/arm q/qd、三腰相对 frame-0 的轨迹增量与 qd、frame
   count/timing/strike，改用
   `configs/a3_upper_stable_stand_v2.json`：
   - 12 个腿 qpos 使用 `AGIBOT_A3_CFG.init_state` runtime stand，腿 qd exact zero；
   - 三腰 q 轨迹按各自 frame-0 做常量平移，使 ready 等于 runtime default 零位，但保留
     动作增量和 qd；
   - root X/Y 和 source yaw 不变，root upright、`z=1.0684 m`；
   - exact A3 重算 body FK/velocity；racket world pose 允许随正确 root 改变，故旧 ball/task
     binding 必须全部重物化；
   - 输出目录 no-clobber、receipt last、三类 authorization false。
   Pod deterministic hold 与训练 smoke 必须证明该 plant birth 至少稳定跨过 `t_hit+margin`。
2. 历史 qvel-only receipt 仍须诚实保留以下已验证事实，但不得再外推成稳定性证书：
   - exact A3 上双脚接触、joint limit 与 unsupported/self-collision 检查 PASS；
     static-contact LP 必须原样记入 receipt；`feasible=null/missing scipy` 不得写成 PASS；
   - 所有 joint qpos、root、frame count、strike frame 不变；
   - 每帧 `right_racket` site position/orientation/linear/angular velocity 不变；
   - 首末 joint/root/body velocity exact zero；
   - 输出目录 no-clobber、report last、三类 authorization false。
3. full scope 不允许此 upper replacement；必须完整重编
   `grounded ready → selected core/window → grounded ready`，再重跑 aim/phase/physical-strike
   binding。旧 fivebind 的 SHA、帧号、旋转和证书不能跨 bytes 继承。
4. 新 upper bundle 先自然跑 `1 env × 2 update`；通过后 fresh 4096-env 只跑五轮定位。必须记录
   mean episode、strike opportunity、raw-hard/table/fall/nonfinite、q_des projection、
   collection seconds、environment-steps/s 和 finite checkpoint。episode 未越过 `t_hit` 或
   strike 仍为零时不得发 long。
5. stable-upper 是 birth/reference/actor 合同修复，无需 old/new 学习 A/B；它的验收是上面
   不变量与行为门。若修复后仍有
   mass raw-hard，才允许一条 birth-only、substep-only、policy-step 末已恢复的 diagnostic
   canary；task phase/current-post edge/nonfinite 继续 Done。
6. upper contact receipt 有两个互斥 schema：历史资产使用 corrected-Z alignment；
   stable-upper 使用整块 contact box retargeted alignment。后者的 authority 必须是
   `a3_stable_upper_selected_rubber_face_center_at_pinned_strike_frame`，必须声明不保留旧
   upper center，并闭合 `retargeted_world_z = ready_root_z + task_z`。不要把 full-motion
   authority 或 legacy 两个 Z 字段混入 stable-upper receipt。
7. N1 producer 的 `SUPPORTED_ACTIONS` 必须指向 stable-upper v2 exact motion bytes，并使用
   与其 `_runtime_site_velocity` 相同的 finite-difference strike speed；不得拿 MuJoCo site
   trace 的近似速度或 v1 bundle 跨 bytes 复用。

stable-upper v2 仍不能用 `qdes=physical_q` 保持到击球窗。动作专属 hold 候选必须在 Pod 的
`hope_isaac_venv` 里生成（系统 Python 没有 exact MuJoCo/HiGHS 环境）：

```bash
/workspace/hope_isaac_venv/bin/python \
  hope_training/whole_body_tracking/scripts/materialize_a3_dynamic_ready_contract.py \
  --action-id <bh_loop_c|bh_block> \
  --motion <absolute-stable-v2.npz> \
  --expected-motion-sha256 <sha256> \
  --stable-receipt <absolute-stable-v2.receipt.json> \
  --expected-stable-receipt-sha256 <sha256> \
  --runtime-contract <absolute-training_contract.json> \
  --expected-runtime-contract-sha256 <sha256> \
  --mjcf <absolute-a3_pingpong.xml> \
  --expected-mjcf-sha256 <sha256> \
  --output <fresh-no-clobber-candidate.json>
```

该工具保留历史 ground-LP feasibility 默认，只为 hold 显式选择按正负可执行力矩归一化的
minimax 目标；它还必须按 exact mapping 在 A3 runtime joint order 与 MuJoCo post-root
actuator order 间 scatter/gather。产物三类 authorization 均为 false。随后用 Isaac
nominal-hold 模式先截取未经 artifact 覆盖的 `raw_env_reset`，再截取
`physical_ready_after_reset_write`、step 1、step 10 和 final/pre-terminal，并持续送同一
hold qdes 到至少 `t_hit+margin`；截图、actual-hard/table/fall 与 root/foot 遥测一起判定
“原生 reset 就怪”“候选 ready 写坏”还是“出生正常但 plant 随后漂移”。通过前不得把候选接进
trainer。
每个动作必须使用 fresh no-clobber 输出，且 Pod 的物理 GPU 先用 NVML/owner lock 只读核对：

```bash
env -u CUDA_VISIBLE_DEVICES \
  HOPE_URDF_IMPORTER_NO_UI=1 \
  HOPE_AGIBOT_A3_USD_PATH=/workspace/franco/runtime_assets/a3_preconverted_usd_1b3fecd7/model.usd \
  LD_LIBRARY_PATH=/workspace/franco/runtime_assets/libglu_af791d1e${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
/workspace/hope_isaac_venv/bin/python \
  hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
  --task HOPE-PingPong-ActionBall-AgibotA3-v0 \
  --num-envs 1 \
  --device cuda:<FREE_PHYSICAL_GPU> \
  --nominal-hold <absolute-dynamic-ready-candidate.json> \
  --nominal-hold-sha256 <candidate-file-sha256> \
  --nominal-hold-receipt-out <fresh-receipt.json> \
  --duration-s <at-least-t_hit-plus-margin> \
  --screenshot-dir <fresh-frame-directory>
```

这是 simulator-only diagnostic：它关闭动作参考偏离终止与随机化，但保留
actual-hard、qdes nonfinite、table 和 fall；任何首个 terminal 立即停止，并把上一安全帧记为
`preterminal`。带截图的 Vulkan/RTX probe 不设置 `CUDA_VISIBLE_DEVICES`，直接把 Pod 物理卡号
写进 `--device cuda:N`；否则 Isaac 4.5 的 GPU/渲染枚举可能不一致。Pod 的 headless importer
还必须显式设置 `HOPE_URDF_IMPORTER_NO_UI=1`。fresh checkout 必须复用按内容钉住的 A3
preconverted USD；否则绝对 URDF 路径变化会触发同一资产的重复转换。当前 Pod 副本四层 SHA
依次为 `1b3fecd7… / 8e521141… / 5b5fc00b… / c76c5bdd…`，private GLU 的
`libGLU.so.1.3.1` 为 `af791d1e…`。日志必须越过 scene creation，不能把
`Simulation App Shutting Down` 的零退出误写成截图证据。正式 table receipt 仍遵守“单卡可见、
logical cuda:0”合同。该 receipt 与截图不授权训练、部署或真机。

nominal-hold 不读取 32 份 `force_matrix_w` 做 table receipt 的 USD/filtered-contact 枚举；该
枚举由独立 formal table smoke 负责。hold 仍保留并逐步读取 `robot_hit_table`、fall、
qdes-nonfinite 与 actual-hard term，因此这不是删除桌碰安全真值，只是把无关的收据物化从 reset
姿势诊断中移走。

以下性能改动若 Pod focused parity 通过，可直接进入 replacement，不另开学习 A/B：

- immutable frozen receipt 的 external SHA cache，不得改变 dataclass/pickle/wire/exact-resume；
- 同一 manager policy step 的 `_compute_strike_timing` 正常路径只算一次，direct/reset 仍重算；
- global + per-action error reduction 保持，`N<=8192` 的布尔 count 改为可精确表示的
  float32 后合成一次
  [device-to-host transfer（D2H，设备到主机传输）](../DEFINITIONS.md#device-to-host-transfer)，逐 step Python EMA、adaptive sigma 和
  reference perturb 逐值不变；
- `fired_valid` 空集保持旧 metric 的 device mask；`exact_strike` no-strike 早退暂不动。

这些补丁仍须在 Pod 跑 focused pytest、fixed-tape parity 和 profiler；“数学等价”免的是学习
A/B，不是回归。只有 4096-env healthy baseline 达到至少 `15k environment-steps/s` 且出现
strike 数据后，才启动 Reward/reference/curriculum 剂量比较。

2026-07-31 的 P0a 已给出本工序的首个强吞吐验收样例。strict 1.1 倍同一 bundle/seed 的
旧 exact `4096×5` wall=`9.00/10.11/25.63/16.81/23.89 s`，新 exact source
`6557390f` wall=`2.90/3.65/18.38/9.70/16.40 s`，均值
`17.088→10.206 s/update`（改善 `40.3%`）；三组深层 update JSON 逐轮相等，五份 checkpoint
全 finite。反手拉 exact source `bd340479` 的同实现 probe 为
`2.90/2.87/12.51/5.97/9.83 s`，均值 `6.816 s/update`，随后才 fresh 发
milestone1000。对应 smoke/probe/milestone spec 应保存在
`configs/n1_speed_hotpath_launch_20260730/`，不能只留 Pod 命令或聊天。

性能比较必须从 bundle bytes/中心来球速度核对实验身份，不能靠 `run_name` 猜。source
`c38b25d0` 的一次 probe 名称虽含 fastball，实际绑定 1.0 倍 bundle，只能记作 spent
构造证据，不得进入 strict 1.1 倍性能比较。不同动作/reset 负载的 wall 也不得混成一个均值；
`6.816 s/update` 证明某一健康窗接近目标，不等于所有动作都已稳定满足健康线。

第三批 replacement 已合并 diagnostic batch birth、compact joint ledger、broker/pool proof
裁剪和 metric/validation D2H packet；随后 `096afb7b / 4d631fb3` 又分别裁掉 diagnostic
外围及内层 rollback 快照，formal/default 仍保持原 rollback 与同步 fail-fast。所有候选的
三组逐 update JSON 与 checkpoint finite 门均通过，但 same-seed collection
`10.3804/10.6618/10.7364/10.2878 s` 均未优于 `10.0916 s` 基线。因此这些提交不得在
发射记录中标成“更快”，只可标成“等价裁剪、性能 FAIL”。

因为 `hope_commands.py` 已改变，必须重钉 profile/bundle/recipe/spec/claim，旧 spec 不得只换
commit SHA。当前 exact 工件为 profile raw SHA=`2c1c91c…9b2c`、base bundle=
`d28a5b12…4246`、strict 1.1× bundle=`81dee53f…0351`、config source=`056625be`。
自然空闲槽固定按 recipe-only → `1 env×2` finite smoke → same-seed `4096×5` 深层
counter/solver parity 与 wall 执行，同时用 CUDA 坏谓词确认 `_assert_async` fail-closed；
当前 long 不热补。现在 probe 已证明仍慢，下一发必须先使用默认关闭、仅 diagnostic
授权的 opt-in 分段 profiler，报告 reset env 分母以及 birth/broker/pool/solver/install
各段。旧 spec 可省略 `diagnostic_update_profile` 并规范化为 `false`；profiler probe 必须在
canonical spec 中逐字写 `"diagnostic_update_profile":true`，使开关进入 launch claim，并由
launcher 在外层 Kit supervisor 与最终 trainer 两层只传
`HOPE_ACTION_BALL_UPDATE_PROFILE=1`。禁止从调用 shell 透传任意 profiler 环境变量。随后把最大段
改为 compact batched reset。不先切 exact table，不优化 PPO。

首个 exact profile 已完成：source `5e1443c4`、policy contract=`0acbbf02…6a57`、
smoke/probe claim=`0b6cfc0c…e80b` / `83898200…8519`。same-seed `4096×5` 中
`solver_solve_many=33.432 s` / 总 collection `51.654 s`，而 install 仅 `0.202 s`；
因此下一发必须先替换 diagnostic receipt 重复验证，不能先做 install packet。profile
只用于归因，性能验收仍必须关闭 profile，并与相同 reset strata 的默认路径比较；三组
逐 update JSON 和 checkpoint finite 门不变。

host-only solver successor 的 canonical diagnostic 输入为 source=`7f77ae5c`、
profile pins=`da4bfd74…172e`、bundle=`c5d8a01f…74b9d`、policy contract=
`b150a395…3703`；smoke/probe claim=`72cc4bb9…a70d` / `7c8f7554…94f63`。
spec 见 `configs/n1_speed_solver_hostonly_launch_20260731/`。它的五轮均值为
`6.700 s/update`，且与 pose-OBB 基线的 reset/行为账本逐轮等价。后续
若改 `cq_n_iters`，必须先在同一 fixed tape 对 `4/6/8/12` 检查 solver
残差、admit mask/reason、racket task 误差与 authoritative replay；这是数值
canary，不是凭训练 reward 均值选迭代数。

#### Reset receipt granularity decision

Move the full human-readable receipt/transcript ceremony out of per-reset hot paths after the
segmented profiler and exact parity checks pass. CC's `(33-4)s / 4100 ≈ 7.1 ms` estimate is an
upper bound per env-reset, not yet a causal measurement; it also contains other ActionBall work.
This is a deterministic implementation optimization and does not need a learning A/B.

The replacement must keep a compact, checkpoint-bounded event journal containing at least
action/env/reset/swing identity, domain epoch/levels, birth↔sample assignment, proposal
admit/reject and reason, the exact GPU float32 task/solver output, lifecycle/outcome, curriculum
generation, and every table/fall/actual-hard/nonfinite truth. `seed + config` alone cannot
reconstruct which env reset when, the policy-caused outcome, or the exact solver task. Launch-time
motion/manifest/solver/physics/Reward SHA and admission gates remain unchanged; full JSON,
historical replay, hash-chain sealing and detailed reports move to checkpoint/hourly materialization.

Acceptance is Pod-only: fixed proposal tape task parity, counter/reason parity, old-receipt
canonical-byte reconstruction from the compact journal, uninterrupted-vs-exact-resume equivalence,
and segmented reset/throughput timing. Do not implement a hash-only journal or use a live mutable
GPU view as immutable evidence.

Diagnostic P0a/lean validation does not close this formal decision. Before formal N=5, the compact
checkpoint-bounded journal and canonical receipt reconstruction above still require exact-resume,
tamper-negative and fixed-workload Pod acceptance. Likewise, the exact 5×32 table-contact backend
stays enabled: its measured fixed cost is only about `0.22 s/update`. A table-frame conservative
box/prism is a backend-failure fallback, not the active speed fix; it must use a finite table region,
collision-geometry center+bound/swept checks, four-substep sticky state and NaN fail-safe rather
than a world-frame infinite half-space or body-origin-only check.

N1 diagnostic launcher 的 budget 名称固定为：

- `smoke`：`1 env × 2 update × save1`；
- `probe`：`4096 env × 5 update × save1`；
- `canary`：`16--1024 env` 的有界 Reward screen；
- [`milestone1000`](../DEFINITIONS.md#n1-milestone1000)：exact
  `4096 env × 1001 update × save100`，自然产出 `model_1000.pt`；
- [`long`](../DEFINITIONS.md#n1-diagnostic-long)：exact
  `4096 env × 20001 update × save100`，自然产出 `model_20000.pt` 的 finite reviewed 长跑预算。

`probe` 只能使用 exact 三元组，不能借 `canary` 或 `long` 填任意值。它仅验收同一 setting 在真实
并行规模下的构造、吞吐、reset 分账和 finite checkpoint，不产生 Reward 胜负或 curriculum
promotion 结论。`milestone1000` 用来买足够长的首轮学习证据，但仍是 diagnostic；它不因运行
更久就自动升级成 formal、curriculum promotion、第二 seed 或真机证据。`long` 同样是有限预算，
不再使用一个不可达的超大 iteration 哨兵来冒充“一路跑”。

### ActionBall finite q_des / reference-reset 切换

本段是 `curr-launch-fix` 功能分支候选；`origin/main/docs/NOW.md` 仍是运行态权威。旧 run 使用
“finite q_des 越包络即 reset”合同，不能 exact resume 成新合同。发射前必须在 effective Reward、
policy/runtime hard contract 和 canonical argv 中同时回读：

1. [`finite q_des execution projection`](../DEFINITIONS.md#finite-qdes-execution-projection)：
   包络内 `executed_qdes == raw_qdes`，有限越界时执行最近合法值，raw action/log-prob 不改；
2. [`qdes_projection_penalty`](../DEFINITIONS.md#qdes-projection-penalty)读取投影**前**的归一化超出量，
   首发 weight=`-5`；`-20` 只允许命名清楚的消融臂；
3. [`reference_guard_mode=metrics_only`](../DEFINITIONS.md#reference-metrics-only)：anchor/body/ee
   只记 counter，不 reset、不额外给 Reward；
4. nonfinite raw q_des、实际关节 current/substep raw mechanical hard edge、table hit 和 fall
   仍各自 hard reset，不能被 reference mode 或 projection 开关屏蔽；仅 predicted ballistic
   crossing 必须生成有限 brake target 而不 reset。实际 q 进入 hard edge 内侧 `2%` 只记
   joint/side/dwell 并由 actual-q barrier 持续收费，不再 reset；q_des clamp 只约束 drive
   target，真实 q 仍可能因隐式 PD、重力、接触和惯性进入该软带。

若 q_des projection/nonfinite 为零、fall 很低，而 `joint_actual_forbidden` 仍让 mean episode
length 长期短于 `t_hit`，先做单变量的滚动刹车 A/B：每个 fresh physics-substep 仍重新读 q/qdot，
但 prediction/brake horizon 保持一个完整 policy/control step（当前 `20 ms`），不能静默缩成
单个 physics tick（当前 `5 ms`）。不得同时改 Reward、Done 或 safety-band margin；只有
1-env smoke 后的 fresh 4096 run 让 episode 越过 `t_hit` 且出现持续 strike，才可替换长跑。

`5e94f21b` 已按上述判据给出反例：updates 1–16 的 actual-joint reset 为
`4,791.6/update`、mean episode `20.19` steps，最近十窗没有相对 `5dbb` 的吞吐提升。下一单变量
改为 [`finite_projection_soft_envelope_inset_fraction=0.05`](../DEFINITIONS.md#finite-projection-soft-inset)：
只在 ActionBall finite projection 模式把 soft q_des 包络上下侧各内缩 `5%`；raw action、
log-prob、Done、actual hard `2%` 安全带和 Reward 权重均不改。该比例必须同时从 action runtime
与 schema-3 training contract 回读；缺字段、config/runtime 不等或旧 checkpoint resume 均拒绝。
必须先用 Pod 1-env smoke，再做 4096 同 seed；source tests 不能代替 Isaac 结果。

`478f485b` 也已给出反例：q_des termination/projection penalty 均为零，但 4096-env updates
0--6 仍约 `4.7k actual-joint reset/update`，mean episode `19--24<t_hit` 且 strike 为零。
不要再增加 q_des inset 或直接放宽 actual band。下一 short diagnostic 必须在每个 PPO update
检查一条 `HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON`，至少含：

- exact articulation `joint_order`；
- episode age `<=1` / `>1` 的 terminal 分母、mean/max age；
- 每个非零 joint 的 current lower/upper/nonfinite-or-invalid、substep actual-hard、
  pre-apply nonfinite q_des 和 predicted-crossing overlap。

这些计数只用于定位，不能晋级 checkpoint。计数器必须在 device 上累加；rollout hot path
不得 host sync，update boundary 才允许一次小批量 D2H。schema v1 旧跑的 terminal 总数与同一
update 的旧 `joint_actual_forbidden` raw count 对账；切换软/硬语义后的 schema v2 必须分别报
`total_safety_event_count` 与 `total_hard_terminal_count`，只有后者与新 termination reason
对账。若不同，先停在诊断，不改训练 setting。

`8d2a1bcd` 已完成旧语义定位：三轮 4096-env 的 event 分母为
`3,187/4,457/5,087`，主因是 left ankle pitch lower，且绝大多数没有 raw-hard overlap。
后继 source 必须先 1-env×2-update smoke，再 fresh 4096 跑至少 3 个完整 update；晋级要求：

- mean episode 的 trailing window 越过 `t_hit≈31`，且 strike opportunity 不再恒零；
- `total_hard_terminal_count == joint_actual_forbidden reason`；
- 2%-inner safety event 可以存在，但 actual-q barrier contribution 必须非零并分账；
- raw hard/table/fall/nonfinite 不得相对 `8d2a1bcd` 增加；
- 报告每 update wall time 和 environment-steps/s，不用更多 env 掩盖 reset。

N=1 launcher 的 canonical Hydra argv 必须逐字包含
`+task.racket.reference_guard_mode=metrics_only`。该键不在 task YAML 中，少写 `+` 会在 compose
阶段失败；不能把 source-level argv 测试当作真实 Hydra 通过。full scope 还必须从 prototype
回读 `full_solver_admission_preflight.diagnostic_gate.status=PASS`，缺该 provenance 的旧 bundle
一律拒绝。

每轮同时记录 per-joint projection trigger、正负侧、投影前 mean/max excess 和执行值恰好贴边的
saturation fraction。触发率下降只能说明候选趋势；冻结 policy、把 projection penalty 置零后复测
仍低，才说明 policy 自己学会了限位。CaT 连续违规调制和 PPO policy-mean bound loss 会改变训练
目标/runner，本轮不临时叠加。

性能判断使用健康对照：旧 `6.4 s collection/update` 来自 mean episode length=`1`、
`98,304 reset/update` 的失败 probe；同代码修正 stand hold 后代表值约 `4.49 s`。旧 v2 也曾把
早期 `1,690–2,301 ee_body_pos reset/update` 学到后期 `1–7/update`。当前旧语义 ActionBall
loop/block 约为 `27/48 s per update`，分别由 ee/qdes 主导；finite-qdes 切换对 block 预计省
`14–17 s/update`，但只有 fresh run 的上面四个 timing 字段能验收该预计。

reference 动作无需因本切换重做：upper/full loop/block 四件老师的 hard/soft/2%-inner crossing
均为 `0`，block 全片 normalized hard/soft margin `0.115081/0.072312` 不小于 loop。该证据只排除
“block 老师贴限”根因，不替代 fresh rollout、table/fall 或实际关节安全检查。

## 冒烟

7. **一格 2-iter smoke**（旧代口径：`4096 env × 24 step × 2 update` full-scene probe，
   或 `512 env × 25 iter` 机制检查）。
   - **让它自然退出**；不许手工 TERM/KILL 探针来伪造"通过"。
   - **通过条件**：`grep -nE 'WARN|Error|Traceback'` 的 WARN 行**全部**进摘要、
     Error/Traceback **零条**；且 `grep -Fc 'q_des CLAMP ACTIVE' > 0`
     （限位剪切 2026-07-06 起默认开，缺这行 = 有人显式关了，只允许出现在老配方复现对照臂上）；
     且 `mean_episode_length` 不恒为 1；若恒为 1，必须按 reason ledger 区分 qdes/reference、
     actual/nonfinite/table/fall 与出生错位，不能再一律写成出生位问题。
   - 一格通过即可发全矩阵（精简治理，不设多层仪式）；高风险波才逐臂解锁。

## 发射

8. **严格串行**：同 pod 内 boot 串行（由 `kit_boot_lock.sh` 持锁保证），
   **看到该臂首个 `Learning iteration` 之后**才轮到同 pod 下一臂。
   两个 pod 可以并行各 boot 一格。相邻两次 launch 错峰 **≥60 s**（同秒启动撞 CUDA 枚举，
   报 "no suitable CUDA GPU"）。
9. **日志目录先 mkdir**（目录不存在 ⇒ 发射壳当场死，连报错都看不到，两犯）。
10. **run_name 当场进实验 run table**；责任/优先级变了才动 NOW。
11. **发射后回读 config**：从 run 的 wandb `debug.log` grep `motion_file` 核对，
    确认 `strike_phase` / `mount_normal_sign` / reward pack 与绑定收据一致。
    **⚠ `motion_file` 路径写错会静默回退到 WandB registry**——不回读就不知道训的是哪条片。

## 监控（只读不写）

12. **首迭代判定**：必须看到 `Learning iteration` **且**绑定的 PID/PGID/starttime/cmdline
    仍是同一活进程。**只有 resume 行不算首迭代**——它会让监视器误报。
13. **boot/stale 双门**：首个 `Learning iteration` 的总 boot timeout 是 `1800 s`，但日志
    连续没有任何推进时，launcher 的 `KIT_BOOT_STALE_TIMEOUT_S=900` 会更早 fail-closed。
    因此“进程仍在高 CPU 构造”不等于可静默等满 1800 秒；若大型 env 构造需更久，必须先量化
    startup phase 并显式修改、记录该合同，不能把 launcher 的 900 秒自然停止写成 PPO 失败。
    出现首个 `Learning iteration` 后仍以 900 秒无推进为 stale。
    task teacher-rate 的边界只有一个真源：
    `canonical_teacher_rate_from_site_speed`。producer 与 Motion consumer 都必须调用它；其
    `5e-7` 绝对容差只吸收 GPU float32 接缝，不会 clip、重定时或放宽 action support。consumer
    不得另写严格区间复验，否则合法边界题会在长跑中随机 Traceback。
14. **里程碑算术**：fresh 要写出 `model_1000.pt` 必须传 `max_iterations=1001`（0 起数）；
    热启动把相对偏移加到父迭代号。**终版存档名是 `model_13599` 不是 `13600`**——
    等 13600 会永远等。
15. **摘要抓异常不抓预期**：WARN 必进摘要。
16. **后台任务卫生**：一个目的一个监视器，目的消失立刻停；**超时参数不生效，必须显式停**；
    每次汇报清点"几个活着、各干什么"。

### N=1 动作专属 dynamic-ready fresh 发射（2026-07-30）

新一代 N=1 upper bundle 使用 schema 2，并把
[动作专属动态准备合同](../DEFINITIONS.md#action-specific-dynamic-ready)的 candidate 与独立
Isaac nominal-hold PASS receipt 一起钉住。launcher 会把
`action_ball_dynamic_ready_bootstrap=true`（物理出生仍是 motion frame 0，但控制目标与 actor
初始输出使用该动作 hold qdes）及两件文件的 path/SHA 逐字传给 trainer；旧 bundle v1 只允许
审计读取，不能进入该发射路径。shared-ready 与 dynamic-ready 不得同时打开，resume 不得覆盖
fresh actor bias。正式写 smoke spec 前，先在 Pod 以 `1 env`、diagnostic 和同一 dynamic-ready
双 pin 运行 `action_ball_policy_recipe_output_path=<fresh absolute path>`；该 recipe-only 构造
只写 exact PPO/policy contract，不做 PPO update。spec 的 `policy_contract_sha256` 必须取自这份
新 schema-2 recipe，旧 shared-ready SHA 不得复用。

fresh successor 还必须把 actor 合同切换为
  [`action_ball_table_pose_twist_heading_task_teacher_start_v2`](../DEFINITIONS.md#action-ball-teacher-start-contract)，
即 exact fixed-194：相对 racket position residual、demanded velocity 和 raw-A face normal 全部
统一到 yaw-heading frame；另加桌面中心 frame 下的 base XYZ、完整连续 6D orientation 与
root-COM 三轴线速度，并直接提供同一 Motion phase governor 的
`time_to_teacher_start_s=max(pre_swing_wait_s-task_age_s,0)`。policy recipe 不直接列
observation 名称或宽度，但 schema-2 dynamic-ready identity 会绑定 candidate/hold 的**绝对
运行路径**与其派生 binding SHA；只要 exact checkout 路径变化，就必须在目标 checkout 上重新
运行 recipe-only 物化，不能复用旧 tracked recipe。旧 182/191-D checkpoint、旧混合-frame
194-D checkpoint，以及当前兼容合同下的
194-D checkpoint 均不能因同宽复用。v2 hard contract 必须由 fresh smoke 的实际 term order/width
和 checkpoint 证明。该 sim
输入由 rigid-body truth 构造；当前 C++ builder 不支持 fixed-194 v2，且真实 marker→base 旋转外参、
gyro 外参和线速度估计器尚未闭合，因此这一步只授权 Pod 训练、不授权真机。

当前 launcher 还逐字加入
`+task.domain_rand.stable_ready_plant=true`。它保留旧 robot-material DR 与 policy recipe
钉住的 joint-default `±0.01 rad`，关闭 torso CoM、link mass 和 PD-gain DR。理由不是方便过门：
旧 loop/block `4096×5` 在第一次 PPO 前分别有 `860/864` 个 env 撞 `waist_roll` raw hard，
hard-env Jaccard `0.982`，而 qdes 与 teacher 都有余量，证明 full DR 的共享 plant 已压过当前
ready 稳定域。fresh `4096×5` 必须验证该 profile 能跨 `t_hit` 且不再 hard 爆炸；1000 update
后再按具名 DR 轴逐项恢复。

验证顺序是 Pod focused tests → `1 env × 2 update` → `4096 env × 5 update` → fresh
`milestone1000`。前两门只判断构造、finite checkpoint、reset 后 q/qdes/last-action 一致，以及
episode 是否能够活到动作 `t_hit`；五轮没有 strike 不能判策略不可学习。进入千轮诊断后在
`200/500/1000` 观察 fatal、finite、teacher imitation、击球机会与真实安全；按历史经验，击球
学习结论至少等到约 1000 updates 和足够 eligible denominator。`milestone1000` 到点后才决定
是否进入 reviewed `long` 或开 Reward/reference/curriculum canary。

旧 diagnostic runner 跳过 formal Reward 时也跳过 joint-safety consumer，却仍每 policy step
生产摘要；4096 槽在约 `170 × 24` policy steps 后必然溢出。fresh successor 必须启用
[diagnostic joint-safety drain](../DEFINITIONS.md#diagnostic-joint-safety-drain)：qdes clamp、
substep q/qdot freshness、brake、raw-hard Done 和逐关节 count/min-gap 全部保留在 device
update aggregate；PPO 前 prepare/validate，optimizer 成功后才 ack/clear。诊断跑不再复制逐
substep dense transcript、逐 policy-step identity，也不再每 update 生成 formal
`.prepared.pt` / optimizer-commit 收据；它始终无 formal 晋级权。已经 sticky overflow 的旧
checkpoint 不得清闩续跑，必须 fresh no-clobber 重发。该 compact 路径必须先在 Pod 通过
fixed-seed parity 和吞吐门，未通过时仍只算 candidate。

## 停止

17. **禁止 `pkill` / `killall` / `pgrep -f` 后批量发信号**——会命中 ssh 远端 shell 或相似 run。
18. 从**经核对的 launch sidecar** 读 exact PID/PGID（不得用命令行模式搜索结果代替所有权 sidecar），
    先保全 checkpoint/contract/log 并核对迭代号/finite/schema/合同 SHA，
    再 `kill -TERM -- "-$PGID"`，仅在 TERM 未退出且证据已落盘时 `kill -KILL`。
19. **通过条件**：exact PGID 消失、认领槽位无本臂 compute PID、`kit_boot_lock` 无 holder、
    其他臂完好、什么都没删。完整流程见
    [RunPod 精确停止](run_on_runpod.md#已登记-phase-1-实验臂的算力释放)。

## 判死之前

- **硬止损**（立即停）：合同/哈希错、NaN/Inf、不可恢复 crash、开关未生效、train/exam 泄漏、
  结果删失或分母错误。先保留日志，不自动换配方重试。
- **证据止损**：至少两个相邻里程碑 **且** 至少一个 `50/侧` 判卷点都显示候选在**较差侧**被对照支配，
  且无其他预注册主指标补偿，才停整对。
- **固定预算 canary 到点自然结束**，它只决定"要不要买第二 seed / 延长"，
  **不能据此宣称 family 永久失败**。
- 同一 `(family, seed)` 的 old/S1 是**不可拆的一对**；除硬失败外不准只停差的一边。
- **跟踪三合格不用来判死臂**（拍面 25° 误差照样 79% 上台的教训）。

## 想把这些变成闸门

第 3、8、9 步现在只是文字。确切检查与落点见
[应当变成闸门的规则](rules_that_should_be_gates.md) P1 #8、#9。

## 带题库的臂:发射清单(2026-07-27 实测,每条都是护栏当场炸出来的)

> 起因见 [EXP-V2-REWARD-FREEZE §5](../experiments/2026-07/EXP-V2-REWARD-FREEZE-20260726.md)。
> 人话:凡是吃虚拟球落点奖励的臂,球拍速度指令**必须是解出来的答案**,不能是盒子里随机抽的。
> 随机抽的话,"听话地照速度走"和"把球打回去"在多数抽样下是反的,回球率注定接近零。

### 0. 先确认题库存在

**这一步最容易被跳过,而它是 2026-07-27 那次"跑不起来"的唯一原因**——两台 pod 上一份题库都没有。

```
PYTHONPATH=<repo>/hope_ws/src/hope_planner \
python scripts/gen_stage1_questions.py \
  --clip <family>:<compiled_clip.npz> \
  --grip off \
  --split train --n 8192 \
  --stroke-guard stats \
  --stroke-budget-clips <fh_v4rg_cal.npz> <bh_v4rg_cal.npz> \
  --out <bank.npz>
```

- `--grip off` 是**硬要求**。默认值 `registry` 会对未注册的 canonical clip 悄悄套一个 40.26° 人手握拍角;
  marker 是唯一权威。
- n=8192 单核约 50 分钟。生成完先看三个数:可解率(应 ≥85%)、torch 闭环落点(应全中、中位 ~3 mm)、
  过网(应全过)。任何一个不对,是**击球点选错了**,不是题库参数问题——挪点,别拿不可解的题去训练。
- 还要看一个泛化量:`answers inside the clip swing-velocity cone`。这个百分比越低,策略要偏离老师越多。
  实测窄带(来球 2–5 m/s)26%,宽带(1.5–7 m/s)21%。

### 1. 发射器必须写全的开关

| 开关 | 值 | 不写会怎样 |
|---|---|---|
| `task.racket.question_bank` | 题库路径 | 护栏拒绝开机(anti-correlated 构型) |
| `task.racket.face_command` | `true` | 护栏拒绝:解出来的拍面没人打分 |
| `task.racket.mount_normal_sign_per_clip` | 每 clip 一个 ±1 | 护栏拒绝:符号错会悄悄判错哪一面胶皮 |
| `task.racket.achieved_target_mix_prob` | `0.0` | 护栏拒绝:HER 回放在题库覆写之前,混进来的目标没解过 |
| `task.racket.vel_range_per_clip` | `null` | 护栏拒绝:题库模式下这是死旋钮 |
| `task.racket.ref_vel_scale` | `1.0` | 同上 |

**hydra 一律用 `++` 前缀**。`+` 在 key 已存在时会炸,`++` 两种情况都对——不要靠猜 key 在不在 yaml 里。

### 2. 拍面观测通道(`face_command_obs`)现在开不了单臂

开它要走 179D 契约 `deploy_parity_face179`,而 schema-3 结构校验**硬性要求正反手双 clip**
(单反手臂给不出 `[+1,-1]`,家族表也要求两族齐全)。单 clip 臂只能 `face_command_obs=false`——
拍面**仍然被主项 `racket_normal` 打分**(权重 0.5),只是策略看不见指令值、得自己从来球推。
双 clip 臂才能开这条通道,那本身就是一条干净的消融。

### 3. 同节点串行起

同一台 pod 上三个 Isaac 进程间隔 2 秒同时起,其中一个在 URDF 导入阶段
`malloc(): invalid size (unsorted)` 堆崩。**等前一个进 Learning iteration 再起下一个。**
