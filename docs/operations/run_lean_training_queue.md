# 轻量 YAML 训练队列

状态：旧机制行已完成机器终态处置，qdot `-5/0` 自然终档，其余历史 namespace 不再发射。两个 P1 配对
现绑定 `main@caeb9ad` 的 clean checkout `/workspace/codexschema/nohope_p1_caeb9ad`；严格 terminal
full-scene probe 已通过，两份队列已显式把 [`launch_authorized`](../DEFINITIONS.md#launch-authorized) 改为
`true`，conditional 与 V1+V2×base-decel 两组科学配对当时均为 `ready`。probe 是非科学启动/终档门，
不是行为晋级，G05 仍为 `Partial`。显式解锁后，conditional control/treatment 已分别在 Pod2
GPU1/GPU2 越过 first iteration（PID=PGID `357023/357679`），paired `model_200` 身份门通过；但
step `180..200` 的 treatment gate/cost/reward 全零，当前 conditional setting 已判
activation-invalid。interaction control
PID=PGID `358331` 则在 first iteration 前的 dynamic URDF import 以 `malloc(): invalid size (unsorted)`、
`rc=134` 自然退出，treatment 未发射；claim/namespace 保全，这不是 interaction Reward 或行为失败。旧
control 行已 `rejected` 且禁止重发；逐字相同配方的新 `control_retry_v2` 与原 treatment 为 `ready`，只能用
同一个 `fill --count 2` 事务在 retry 越过 first iteration 后再发 treatment。该事务已按序完成：retry-v2
PID=PGID `359240` 在 Pod2 GPU1、treatment PID=PGID `359872` 在 GPU2，二者后来均自然到
`model_1000` 并退出；finite/fresh/contract 终档身份通过，但旧 source 的 V1/V2/base-decel
denominator/numerator 不全，只记 instrumentation-blocked，不解释 Reward 效果。新 replacement 队列已绑定
history-reachable exact successor `0f3900a612863faf326dca6ad3e8d38bfe8df3c9`：它令 control/treatment 都在
RewardManager 同一阶段执行 weight=`1.0`、返回严格零的 probe，且与真实 base-decel term 共用
kernel 参数与 step-token 去重。两份队列仍保持 `launch_authorized=false` / `blocked`；必须先水合该
source 的 ignored A3 资产，用该 exact source 自己重跑 strict full-scene terminal probe，再由显式 consumer
解锁。不得复用 caeb receipt 或旧 pair 曲线。

qdot matched-control 的首次冷启动也在 dynamic URDF import/scene creation 前停住；相同 warning 在成功臂
同样存在，不能拿 warning 字面当根因。旧 namespace 已保全并拒绝，只有 unchanged retry-v2 可再试一次。
launcher 现有默认 180 秒“有内容但 size/mtime 均不再变化”watchdog：任一增长重置计时，marker 在边界
poll 上优先。spawn 后由 source-pinned `exact_process_group.py` 双读 leader 的 `/proc/<pid>/stat`
`starttime`，并交叉核对 `getpgid` 与 `PID=PGID`；sidecar 绑定 adjacent leader evidence。超时前再次双读
同一 identity，枚举并双读该 exact PGID 的成员，再写 pre-TERM evidence 后才允许 TERM。TERM 后不再只看
leader：轮询整组；只有 residual 是 pre-TERM 成员的 exact 子集时才允许 KILL。PID reuse、leader 在 TERM 前
消失、成员后来加入或 identity 漂移均绝不 signal，记录 `manual review` 并以 rc121/122 fail closed。正常
stale 返回 rc125，空日志仍由 900 秒 hard timeout 管，stat 失败 rc126。仍缺 scene-created 等细分阶段 receipt；长期应
消费内容绑定的预转换 USD，避免每条训练重新动态导入 URDF。

### 新 source 先做 1-env 缓存探针

动态 URDF importer 会在 reward/contract/PPO 之前偶发卡死，所以新 source 不再用正式科学 namespace
试冷缓存。[`boot-warmup`](../DEFINITIONS.md#boot-warmup) 从一条已预注册 job 派生完全独立的
`_boot_warmups/<source>/<pod>/gpu<N>/<attempt>`：动作/bank/exam/plant/seed 仍绑定，但 harness 强制
`1 env × 2 updates`、save interval `1`、独立 schema-2 claim 和 180 秒总 boot 上限。它使用专用确认 token，
不能被科学 launch token 误授权；输出明确写 `not_science=true`。

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml \
  boot-warmup \
  --job-id fresh_c_conditional_face_matched_control \
  --pod pod2 --gpu 1 --attempt-id conditional_61007e9_gpu1_a1

python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml \
  boot-warmup \
  --job-id fresh_c_conditional_face_matched_control \
  --pod pod2 --gpu 1 --attempt-id conditional_61007e9_gpu1_a1 \
  --execute --confirm SIM_ONLY_LAUNCH_ONE_BOOT_WARMUP
```

先看 dry-run。已有 attempt directory 一律拒绝覆盖；reserved Pod 不在 `dispatch_pods` 时也会在 SSH 前拒绝。
出现 `Learning iteration` 只证明 importer/scene/simulation 能走通，仍须等 warmup exact PID 退出并核对无
fatal，才可启动正式 control/treatment。warmup 的模型、Reward 与 checkpoint 一律不得进入成绩表。

首次真实执行 `conditional_61007e9_gpu1_a1` 已按上述路径自然退出，2/2 updates、`model_0/1` finite、
contract/claim/fresh lineage 匹配且 fatal0；该 receipt 只证明此 source/Pod2/GPU1 的 boot 路径。

### 新 source 的 ignored A3 资产先显式水合

clean detached source 不包含 Git 忽略的 A3 URDF/mesh tree。P1 source 行因此可选声明
`source.ignored_runtime_asset`：target checkout 相对路径、donor checkout/commit/相对路径、完整 `46` 文件/
`15,378,264` bytes/canonical tree SHA，以及“禁 symlink、target 必须 Git-ignored”两条硬规则。旧历史行不声明
时保持兼容；声明者的 schema-2 science claim 会自动绑定完整 source mapping，不能只绑定 commit 后另手抄资产。

先 dry-run，再只对 `dispatch_pods` 中用户明确选择的一台 Pod 执行：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml \
  prepare-source-assets \
  --job-id fresh_c_conditional_face_matched_control_p1r1 --pod pod2

python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml \
  prepare-source-assets \
  --job-id fresh_c_conditional_face_matched_control_p1r1 --pod pod2 \
  --execute --confirm SIM_ONLY_PREPARE_ONE_LEAN_QUEUE_SOURCE_ASSET
```

该入口不读 GPU、不启动 Kit，也不访问 reserved Pod1。它在 source-specific lock 内要求 source 没有 exact
`scripts/train.py` 进程；先复核 source/donor clean exact、donor 46-file tree 和 URDF 的 43/43 个唯一 mesh
引用。target 缺失时只复制到 source 外的 deterministic no-clobber staging，复核后用 Linux
`renameat2(RENAME_NOREPLACE)` 原子发布；target 已存在时只允许 exact idempotent verify，不会重拷或覆盖。
完成后在 source 外写 deterministic no-clobber receipt。SSH timeout 是 `UNKNOWN`，命令没有自动 replay；残留
staging 必须保全诊断。

声明该合同的 `doctor`/science launch 会在 Hydra compose、run directory、claim 与 Kit 之前重算 donor/target、
确认 target 仍 Git-ignored，并**消费已有 exact receipt**；doctor 不会替 prepare 补写 receipt。target/receipt
缺失、额外文件、内容漂移、symlink、特殊文件、donor commit 漂移或 43/43 mesh closure 不完整均 fail closed。

### 正式环境数先做 full-scene probe

1-env 成功不能代表 4096-env scene 可创建。[`full-scene-probe`](../DEFINITIONS.md#full-scene-probe)
必须从同一份 ready/blocked exact job 派生：source、目标 Pod/GPU、base+delta、动作、bank、exam、seed 和原
`num_envs` 全部不变，命令没有 `--num-envs` 逃生口；只把 `max_iterations`/save interval 改为 `2/1`，并换成
`full_scene_probe_not_science_*` run name 与
`_full_scene_probes/<job>/<commit>/<pod>/gpu<N>/<attempt>` namespace。先 dry-run：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml \
  full-scene-probe \
  --job-id REPLACE_WITH_READY_OR_BLOCKED_EXACT_JOB \
  --pod pod2 --gpu 1 --attempt-id full_scene_a1

python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml \
  full-scene-probe \
  --job-id REPLACE_WITH_READY_OR_BLOCKED_EXACT_JOB \
  --pod pod2 --gpu 1 --attempt-id full_scene_a1 \
  --execute --confirm SIM_ONLY_LAUNCH_ONE_FULL_SCENE_PROBE
```

专用确认词不能与 science/warmup token 混用；blocked 行也先按 ready 级规则拒绝 zero commit、placeholder、
非 `/workspace` 或重复资产。声明了 `preferred_slot` 或
[`required_slot`](../DEFINITIONS.md#required-slot) 的 job 只能在绑定槽探测，reserved Pod 在 SSH 前拒绝。
execute 的外层容量预检只 SSH 读取用户明确选择的 dispatch Pod/GPU，不调用普通 fill 用的 all-Pod
`live_snapshot`；随后远端仍在同一 selected GPU 的 fd8 短锁内复核容量。这样 Pod2-only probe 不会因防重复
逻辑触碰 reserved Pod1；普通 fill/claim 现在也只读取 `dispatch_pods`，交接前靠历史行终态化而不是访问
reserved Pod 防重复。
远端在 fd8 短锁内重查容量，用 plain `mkdir` 和 no-clobber
`full_scene_probe_claim.json` 拒绝复用 attempt；显式使用 source-pinned launcher 的 900 秒 hard timeout、
180 秒 content-stale watchdog 与 `Learning iteration` marker。
发射资格还要求 row 的 `runtime_binding=true`，并显式绑定 face179 actor observation、31/31 零摩擦 intent 与
物理球。source 上的 probe runtime、queue runtime、trainer callback 定义/调用存在性会在 GPU 容量检查、
`mkdir` 和 claim 写入之前完成；能力缺失不得留下半个 attempt namespace。

launcher 返回只表示 `first_iteration_observed=true`，不表示两次 update 已终档，更不表示 reward 有效。
probe 永远 `not_science=true / attestable=false / promotable=false`，不用 science 的 `queue_claim.json`、
`run_binding.json` 或 `milestones/`。它改用独立的 `full_scene_probe_claim.json`、trainer-owned
`full_scene_probe_binding.json` 和只自然 `wait`、绝不发 signal 的 supervisor；launcher 仍只报告 started/
first iteration，不负责终档或解锁。claim 还绑定 exact supervisor argv prefix、fresh lineage=1 与 Pod-specific
ignored-runtime-asset receipt；任意替换 wrapper 或 causal lineage 都不能解锁。

trainer 自然退出后必须另跑终档器；先 dry-run，再只访问显式选择的 Pod：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml \
  finalize-full-scene-probe \
  --job-id REPLACE_WITH_READY_OR_BLOCKED_EXACT_JOB \
  --pod pod2 --gpu 1 --attempt-id full_scene_a1

python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml \
  finalize-full-scene-probe \
  --job-id REPLACE_WITH_READY_OR_BLOCKED_EXACT_JOB \
  --pod pod2 --gpu 1 --attempt-id full_scene_a1 \
  --execute --confirm SIM_ONLY_FINALIZE_ONE_FULL_SCENE_PROBE
```

终档器不扫描另一台 Pod、不发 signal、不改 YAML status、不自动 retry。它先要求当前 YAML 重算的 expected
claim SHA 等于 immutable claim。queue shell 仍先跑一次 source-asset doctor 作为快速诊断，但终档授权不信任该输出：
probe runtime 会再从 claim 导出唯一 receipt、target 和 donor，自己重算两棵当前文件树库存与 URDF mesh
引用闭包，复核 donor clean commit，并将 target/donor 的实测 tree SHA/字节数/文件数/闭包写入
immutable result 的 `source_asset_receipt.current_closure`。因此直接调用 runtime finalizer、或 doctor 通过后资产发生
漂移，都不能解锁。trainer/supervisor 任一
仍存活或原 PGID 还有 orphan 时只报 not-ready，不写结果；整组自然消失后才核对 exact PID/starttime、normal
rc0、scene→contract→first-iteration phase、fatal0、`model_1.pt` filename/embed/finite/fresh-lineage1、相邻
schema-3 hard-contract SHA、launch claim、source/ignored A3 closure 与 motion/train-bank bytes。通过或失败都
no-clobber 写 `probe_result.json`；controller 明确解析 `terminal_status/unlock_authorized`，失败恒为
`unlock_authorized=false / automatic_retry_authorized=false`。重复终档只接受逐字节相同结果。该结果仍不能进入
普通 milestone attestor 或成绩表；未来显式 unlock consumer 只能消费 `status=passed` 的 exact receipt，当前
queue 不会自动解锁任何科学 job。

终档 pass 还要求 `scene_import_done` 报告的 **实际** environment 数等于 claim、物理球已启用且球/碰撞桌/
视觉桌三个实体都存在，hard contract 为 face179 且 31 个 PhysX friction coefficient 全零；schema-3 validator
从 claim-bound clean checkout 直接 file-load，不依赖 Kit package import。若没有 supervisor exit receipt，但
launcher 已在精确 PGID 收口后发布 pre-marker/watchdog/stale/boot-timeout 终态，终档器只会保存 immutable
failure，不能 pass。CLI 不需要也不应手传 source-asset receipt；路径由 immutable claim 导出。

旧 `main@c7e1a90` 已跑过一次非科学基础设施 canary：`probe_result.json` 内容 SHA-256 为
`02780b52df27255eea096f34dda9a26e806ae3a196c233a46a2af1cde16c4186`，`model_1.pt` SHA-256 为
`a813ea9ba8c058cf5ed2f9a9a8f8fe3b95ec0903cd3702831b99736736738e68`，相邻 hard-contract SHA-256 为
`c39cf1ae4bd99aa5ddce2a4c6c51cfd3858eba4884baeb369d5fdb1cf88df838`；76 个 tensor 中
1,762,715 个浮点元素全 finite，fatal 命中为 0，trainer/supervisor 的原 PGID 自然为空。旧结果内的
`unlock_authorized=true` 只符合 c7 旧终档语义，**不能**解锁当前队列。

严格 `main@caeb9ad` probe `caeb_strict_terminal_pod2_gpu1_a1` 随后通过：result/claim/model/hard-contract
SHA-256 分别为 `0d03bd0305a56e56440b14e1f41278a26c0cad3a84cc1245325faed1ef29b1d1`、
`7437db488d8aa062aba8de91fb517362cc609a81900f0e953f80e15174c36ad5`、
`e1b79d142c13bc2df513b2a7311fbeb7b610fc64047e095c1a54c76571fe3106`、
`c39cf1ae4bd99aa5ddce2a4c6c51cfd3858eba4884baeb369d5fdb1cf88df838`。result 绑定 clean caeb source、
实际 4096 environments、`physical_ball=true`、三实体全存在、76 个 tensor / 1,762,715 个浮点元素全
finite、fatal0 与自然空 PGID，因此 `unlock_authorized=true` 可被显式队列变更消费。probe 本身仍
`not_science=true / attestable=false / promotable=false`，不能当 Reward 结果或晋级证据。

2026-07-16 的 rolling-timing source `704bf3a2` 又暴露一个 supervisor 身份竞态：4096-env×2-update
训练自然 rc0、finite `model_1`、schema-3/fresh lineage、fatal0 与自然空进程组均独立通过，但 child
Popen 后第一次 `/proc` 读取瞬时失败，旧 supervisor 便把 `starttime_ticks=null` 固化进 exit receipt；strict
finalizer 正确拒绝了它。修复后的 supervisor 在最多 2 秒内只读重采样，只有 PID/PGID/正整数 starttime
和 post-exec argv 全匹配才接受；child 早退或超时仍发布不可通过的身份错误，绝不 signal。旧 a1 failure
保持不可变、不会因代码修复被追认或自动重跑。

该 caeb attempt 在当时的 queue shell 中确实完成了 source-asset doctor 重哈希；但它使用的旧 runtime
并没有 in-process `current_closure`，所以不把新能力倒算到旧 result。新 source 的验收结果必须实际携带上述
当前闭包证据。终档 checkpoint 的内嵌 iteration 与 fresh lineage 只接受 plain integer，`true` 不等于合法的
`1`。当前 full-scene terminal 专项 `39 passed`，包含直接绕过 wrapper 后分别篡改 target/donor、以及 boolean
iteration/lineage 的负测；整合 harness/source-asset 回归 `146 passed`。

这个入口解决的是“动作和题库已经决定后，为什么还要手拼一长串命令”。一条 YAML job 必须同时绑定：

- 一个动作名与它的一个或多个 motion 文件；
- 该动作专属的训练 bank 和 immutable exam；
- 一个 clean Git source commit；
- 公共 base recipe 与本臂唯一 delta；
- seed、环境数/迭代预算、`+200/+500/+1000` checkpoint milestone；
- 一个显式 [`dispatch_pods` 可发射 Pod 集合](../DEFINITIONS.md#dispatch-pods)与 GPU round-robin、
  [`preferred_slot`](../DEFINITIONS.md#preferred-slot) 或
  [`required_slot`](../DEFINITIONS.md#required-slot) 资源策略。

它是执行清单，不是第二份优先级账本。job 顺序必须抄自
[`NOW` 统一队列](../NOW.md#统一工作队列唯一优先级账本)中已经解锁的项目；新科学问题仍先写对应实验记录。
示例文件 [`lean_training_queue.example.yaml`](../../configs/lean_training_queue.example.yaml) 故意保持
`blocked`，不会启动 trainer。

## 启动前只查什么

探索训练先在本地把 recipe 编译成单义的 Hydra `key=value` 列表：同一个 key 即使分别写成 `key`、`+key`
或 `++key` 也只能出现一次；recipe 不能改写 seed、预算、run name、device、motion/bank binding 或 launch
claim，且拒绝 Hydra flag、删除语法和 `${...}` interpolation。所有 `run_dir` 在整份 YAML 内必须唯一，
不能等于或位于自身 source 内；`ready` job 还不能放进其他 ready source checkout。这些错误都在 SSH 前失败。

远端快速检查为：source checkout 处于 YAML 记录的 commit 且 clean；若声明 ignored runtime asset，则先消费
exact hydration receipt 并复核 donor/target closure；motion/train bank/exam 三类资产存在；
同一个 child environment 下 `whole_body_tracking` exact 解析到该 source；用最终训练 override 向量实际执行
`train.py --cfg job --resolve`，只做 Hydra 配置合成而不启动 Kit；目标 GPU 未达容量；以一次原子 `mkdir`
创建全新的 run directory；Kit 经现有 boot lock 串行启动。
doctor 不做逐文件 SHA、`pip freeze` 或通用 import closure。真实 queue trainer 另有一个窄的运行绑定：
它只把 claim、真实进程与 exact RSL log directory 绑定，并在预注册 checkpoint 处产生里程碑 receipt；
这不替代正式晋级、跨引擎判卷或 Gate3 的更严格资产/runtime closure。

`setup_train_env.sh`、`scripts/train.py` 与 `scripts/launch_kit_training_locked.sh` 三个入口已固定为 source
checkout 下的 canonical repo-relative 路径，YAML 不能替换成绝对路径、`..`、broad process-control 或
robot runner。`ready` job 还会在任何 SSH 前拒绝全零 commit、非 `/workspace` 路径、placeholder、`..` 和
重复 input identity；blocked 示例可以保留尚未填实的占位值，但永远不会被调度。run directory 不是
“存在就复用”：任何已有目录、文件或 symlink 都会让原子创建失败，因此旧日志和 state 不会被覆盖。

Pod 容量上限固定为 Pod1 每卡最多 4 个本项目 trainer、Pod2 每卡最多 3 个；但真正可发射和读取的机器都由
`dispatch_pods` 显式收窄。2026-07-14 起 active queue 设为 `[pod2]`，Pod1 全留给 Yikang：普通
`status/doctor/fill/launch-next/attest` 的 live snapshot 只 SSH Pod2，不再以跨 Pod 防重为由读取 Pod1。
因此同一变更也把所有历史 `ready` 行改为 `complete/rejected`，新 P1 使用从未发射的 run namespace；不能
单独恢复旧 `ready` 状态。Pod2 按 `gpu0 → gpu1 → gpu2` 完成一整圈，才给任一卡放第二条；真实 launch 前用
`nvidia-smi` 把其他人的 compute process 也保守计入占用。并发 `launch-next --execute` 先竞争单控制端全局
scheduler `flock`；只有持锁者才重新读取 Pod2 三卡、跳过已有 claim、做 round-robin 选槽并启动，因此同一
控制端的多个 agent 不会基于同一份旧快照抢同一槽。`nvidia-smi` 偶尔会为同一 trainer PID 返回重复行；live
snapshot 与远端最后容量检查都按每 GPU 的唯一纯数字 PID 计数，重复行不能把一条 trainer 算成两条。
远端每 GPU 的 boot lock 仍作最后一道容量/claim 检查。

旧实现用 `flock FILE command` 包住启动，flock 的私有 fd 会被 detached trainer 继承，导致一条长训把
该 GPU 的 launch lock 持有到终档，配置容量 3 实际退化成 1。现改为短命 controller 显式持有 fd8，并在
调用 launcher 时 `8>&-`；trainer 不再继承，锁只覆盖 doctor→容量→claim→boot marker。现役旧 trainer
已经继承的锁不强行剥离，只等自然退出。job 可选
[`preferred_slot`](../DEFINITIONS.md#preferred-slot)；槽未满时优先，同卡满后自动回到 round-robin，不能突破容量或
`dispatch_pods`。需要**绝不 fallback** 时改用
[`required_slot`](../DEFINITIONS.md#required-slot)：槽满就不分配，且本 job 不会落到其他 GPU；它不阻塞
其他槽可运行的独立 job。两种字段互斥。它不保证多 job 原子发射，matched
pair 仍须使用 fresh namespace，并由一次 `fill` 的顺序事务保证第一臂 embedded preflight/首 iteration 失败时
不继续第二臂。science claim、boot warmup、full-scene probe 与 finalizer 都在 SSH 前强制 required 绑定；
probe/warmup/finalizer 还保持 preferred 证据槽绑定。

`doctor` 与 trainer 共用一个 child-environment builder，同时设置所选 CUDA 和
`PYTHONPATH=${HOPE_WBT_PYTHONPATH}`，并共用同一条最终 training argv。exact module probe 和 no-Kit Hydra
compose 都位于 `mkdir/claim` 之前；失败不会污染新 namespace。claim 的 canonical content 自动绑定 source
commit/checkout、完整 caller argv、run name、seed/预算/milestones、motion/bank/exam identity、Pod/GPU；
其 digest 作为 `++training_launch_claim_sha256=<sha256>` 自动加入 compose 与真实 trainer argv，完整执行 argv
也写入 `queue_claim.json`。只有 exact source 含 P1 callback/runtime 且 job 明确设置
[`runtime_binding: true`](../DEFINITIONS.md#runtime-binding-flag)时，harness 才注入 claim/binding absolute
path；trainer 选定真实日志目录后
原子写 [`run_binding.json`](../DEFINITIONS.md#trainer-run-binding)，外部代码不得再按 timestamp/glob 猜
checkpoint 所属目录。这里的 input identity 是路径与语义绑定，不冒充文件内容 SHA。
pending claim 在 NVML 尚不可见时作为 GPU reservation，terminal/rejected 旧 claim 不占新槽。
真正批量发射只用一个 `fill` 进程；它持 scheduler lock，每臂只发出**一次**远端原子 launch SSH。该
launch 自身先在远端短锁内完成与 standalone `doctor` 相同的 source/assets/module/Hydra checks，再做容量、
`mkdir`、claim 与 Kit spawn，并等到第一个 `Learning iteration`；不存在先发一次 standalone doctor、随后
在 launch 内重复全套 preflight 的第二次 SSH。之后才重新读取 claims/GPU 决定下一条。execute 返回
`result_schema_version=2`，每条结果以 `preflight_mode=embedded_in_atomic_launch` 明示检查来源，不伪造已经
删除的 `doctor_output`。独立 `doctor` 命令和 dry-rendered doctor/launch 仍保留给人工诊断。不要并发调用
多个 `launch-next --execute`。

## 命令

`plan/status/doctor` 默认不连 Pod。`launch-next/fill` 还要求机器清单显式
`launch_authorized: true`；通用 example 仍保持 false 并会在任何 SSH 前拒绝，当前两份 P1 active queue
已经 strict probe 后显式解锁：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/lean_training_queue.example.yaml plan

python3 scripts/run_lean_training_queue.py \
  --queue configs/lean_training_queue.example.yaml status

python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml doctor

python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml plan

python3 scripts/run_lean_training_queue.py \
  --queue configs/lean_training_queue.example.yaml \
  attest-milestone --job-id REPLACE_WITH_JOB_ID --milestone 200
```

`plan --live` / `status --live` 只对 `dispatch_pods` 中的每台机器建立一个低频只读 SSH；当前两队列因此
只读 Pod2 的 GPU 与 claim。`doctor --live`
不创建 run directory、claim 或 Kit 进程；它验证 source/assets/exact module origin，并以真实最终 override
向量执行 `train.py --cfg job --resolve`。只有返回 `hydra=exact-no-kit-compose` 才通过。当前 terminal probe
与显式 unlock 已完成，仍须先看 `fill` dry-run，再用单个 scheduler 进程发射指定上限：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml doctor --live

python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml fill --count 1 \
  --execute --confirm SIM_ONLY_LAUNCH_ONE_LEAN_QUEUE_JOB
```

`fill` 最多消费 `--count` 条，并在每条 first iteration 后重采现场；每条 execute 只有一次包含内置 doctor
门的远端 launch transaction，`launch-next` 仅保留单条诊断用途。
`blocked`、`complete`、`rejected` 永远不会进入候选；
claim 写入前 run directory 必须由本次 launch 首次创建。预检或 Kit 启动失败后已创建的 namespace/claim
会保留，禁止自动重试，先诊断再新建明确的后续 job。工具不包含信号、
`pkill`、`killall` 或任何真机入口。

## 里程碑取证

只有由含 P1 callback 的 exact source、以 `runtime_binding: true` fresh 启动并成功写出 immutable binding
的 job 才能取证；默认 false 的旧 source/boot warmup 不受影响，旧 run 不会被追认或补造 binding。当前
YAML row 会在 execute 前与远端 immutable claim digest 重构对比，source/recipe 漂移会在 SSH 前拒绝。
默认 dry-run 只显示绑定/receipt 路径与远端脚本，不连 Pod：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml \
  attest-milestone --job-id REPLACE_WITH_JOB_ID --milestone 200
```

写远端 receipt 需要独立确认词，且 Pod 只能从已存在的 immutable queue claim 解析，不能由调用者另传：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml \
  attest-milestone --job-id REPLACE_WITH_JOB_ID --milestone 200 \
  --execute --confirm SIM_ONLY_ATTEST_ONE_LEAN_QUEUE_MILESTONE
```

[`milestone attestor`](../DEFINITIONS.md#milestone-attestor)只沿 binding 的 exact log directory 打开
`model_200.pt`；它核对静态/读中 PID reuse、binding 时 source HEAD/clean、filename/embed iteration、所有
floating/complex tensors finite、相邻 schema-3 hard-contract SHA 与 launch-claim lineage，再 no-clobber
写 `milestones/model_200.json`。它不运行 judge、
不产生成绩，也不自动停止/晋级。完整字段与失败语义见
[运行绑定接口](../interfaces/lean_training_run_binding.md)。

## 冷启动/URDF import 诊断

queue trainer 默认关闭的 phase telemetry 在 P1 run 中依次留下 `hydra_resolved`、`app_started`、
`log_dir_bound`、`scene_import_start/done`、`hard_contract_written`；冻结 launcher 看到真实
`Learning iteration` 并成功返回后，
queue harness 才在 `.launch` 写 `phase=first_iter`。这能把“卡在 A3 import”与学习失败分开。

## 10000-update 无随挥回放三格漏斗（2026-07-15）

这批只允许 Pod2 GPU2：普通对照与两个 treatment 共用 source/题库/seed；一个 treatment 增加关节速度
边界惩罚，另一个放松击球窗动作模仿。先做 dry-run/doctor，再发三条：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_long_no_replay_funnel_20260715.yaml plan
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_long_no_replay_funnel_20260715.yaml doctor --live
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_long_no_replay_funnel_20260715.yaml fill --count 3
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_long_no_replay_funnel_20260715.yaml \
  fill --count 3 --execute --confirm SIM_ONLY_LAUNCH_ONE_LEAN_QUEUE_JOB
```

三条 job 都有 `required_slot: pod2/gpu2`；不得删除硬槽位来追求表面利用率。任一条没有越过首个
`Learning iteration` 时，顺序 `fill` 会停止，不会自动发后续臂或重放失败 namespace。

## Pod2 两张空卡的六格长曲线（2026-07-15）

当 live snapshot 证明 GPU0/GPU1 没有 Yikang 或其他 compute PID 后，使用
`phase1_long_scaleout_funnel_20260715.yaml`。行顺序是 GPU0→GPU1 逐圈各一条，不先塞满单卡：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_long_scaleout_funnel_20260715.yaml plan
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_long_scaleout_funnel_20260715.yaml doctor --live
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_long_scaleout_funnel_20260715.yaml \
  fill --count 6 --execute --confirm SIM_ONLY_LAUNCH_ONE_LEAN_QUEUE_JOB
```

六格补齐单独/组合模仿、关节速度剂量和脚部朝向剂量，不是六个 seed。它们全部到 10000 updates；
200/500/1000 只判 fatal、finite、合同与可观测 activation。真实击球后才有收入的稀疏结果在最少
eligible hit denominator 未满足时一律继续。首次 V2-only 在 importer rc134 后保全，队列只含一个
逐字同配方的 retry-v2；同 phase 再失败时不得继续造 namespace。

## Pod1 十二格连续平衡与击球 Reward 长曲线（2026-07-15）

Pod1 获重新授权且 live snapshot 证明三卡都没有 compute PID 后，使用
`phase1_pod1_long_balance_reward_grid_20260715.yaml`。队列顺序严格按 GPU0→GPU1→GPU2 四圈，
每卡最多四条；前六条是非击球臂模仿开关 × 10/16/24 秒 episode，后六条是位置/速度/拍面 Reward
的不同配比或总强度，不是重复 seed。

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_pod1_long_balance_reward_grid_20260715.yaml plan
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_pod1_long_balance_reward_grid_20260715.yaml doctor --live
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_pod1_long_balance_reward_grid_20260715.yaml \
  fill --count 12 --execute --confirm SIM_ONLY_LAUNCH_ONE_LEAN_QUEUE_JOB
```

发射前只需一次 selected-Pod asset receipt、12 行 live doctor 和一个同 source/scene/4096-env 的非科学 probe；
不为每条重复做 importer probe。若 probe 或 trainer 在首迭代前出现 importer malloc，保全 namespace/claim/log，
同一 recipe 最多一个 fresh retry；第二次同 phase 失败转根因线。稀疏回球结果在 eligible opportunity 不足时继续到下一正式
milestone，不得用零收入早停。

## Demo-only model-3500 严格续训（2026-07-16）

generic lean queue 继续只接受 fresh run。为了次日演示而允许显式合同变化的续训只走专用入口；七条后代
永久 formal-ineligible，不得用 generic token 发射：

```bash
python3 scripts/run_phase1_demo_hotstart_queue.py \
  --queue configs/phase1_pod2_demo_hotstart_portfolio_20260716.yaml plan

python3 scripts/run_phase1_demo_hotstart_queue.py \
  --queue configs/phase1_pod2_demo_hotstart_portfolio_20260716.yaml parent-inspect

python3 scripts/run_phase1_demo_hotstart_queue.py \
  --queue configs/phase1_pod2_demo_hotstart_portfolio_20260716.yaml parent-attest
```

两条命令默认都是 dry-run。先用确认词 `SIM_ONLY_INSPECT_DEMO_WARMSTART_PARENTS` 执行只读 inspect；它读取
三个 live model-3500、相邻 hard contract、原始 queue claim 和 run binding，验证 canonical content SHA、完整
argv/source/run/process、actor+critic、非空 optimizer `state/param_groups`、finite、schema-3 及双 SHA lineage，
但不创建 snapshot/receipt。随后 `parent-attest` 用确认词
`SIM_ONLY_ATTEST_DEMO_WARMSTART_PARENTS`；它会自动再做一遍 inspect，只有通过才以 `O_EXCL` 写固定只读
`parent_snapshots_v2` 四件套并从 snapshot bytes 复核、写 v2 receipt。旧 receipt 路径永远不接受，不自动激活
或重试。operator 必须把 receipt file SHA、三个 checkpoint/hard/claim-file/claim-content/binding-file/
binding-content/launch-claim SHA 回填 YAML，再把 activation state/status 作为一次受审变更切到
activated/ready。未回填时 `fill` 恒 fail closed。

激活后先 dry-run；真实发射使用本专用 token：

```bash
python3 scripts/run_phase1_demo_hotstart_queue.py \
  --queue configs/phase1_pod2_demo_hotstart_portfolio_20260716.yaml \
  fill --count 9

python3 scripts/run_phase1_demo_hotstart_queue.py \
  --queue configs/phase1_pod2_demo_hotstart_portfolio_20260716.yaml \
  fill --count 9 --execute --confirm SIM_ONLY_LAUNCH_ONE_DEMO_WARMSTART_JOB
```

execute 会只从 fixed parent snapshots 重算三个 parent 并要求与 immutable receipt 逐字节一致，再读取 Pod2
现场容量；live 母本后续变化不会产生 TOCTOU。前六个 job 只硬绑
GPU0/GPU1、按 0→1 逐圈。六个旧 model500 都保全且两卡 occupancy 各 `<=3` 时，先把 job1/job2 放入
各卡第 4 槽；其余四条等待四个弱臂按其证据和 exact PGID 退出后再补，保留 GPU0 V1-only 与 GPU1
foot-`-0.6`，最终各四条。第七条硬绑 GPU2 第四槽，使用 16 秒 episode 观察同回合 3–4 拍的平衡债；
其 claim 额外绑定 `+200` 只判结构/激活、`+500` 判安全/平衡、`+1000` 排候选，且稀疏命中为零不可早停。
不抢占、不 signal。claim 同时绑定 parent checkpoint/
hard/原始 claim/binding SHA、activation receipt、完整 argv 与 `formal_exact_eligible=false`。source 中两个
负责 strict resume/hard-contract 的文件还会按固定 SHA 验证。launcher 只有在 run log 包含 explicit mismatch、
snapshot 路径、iteration 3500、`optimizer=resumed`，且新 hard contract 的 qdot/conditional-face 值与该行相同
后才继续。就在同一个 GPU fd lock 内且 trainer 前，四个 parent snapshot file SHA 还会再与 activated queue
对拍；任何漂移都在创建 run_dir/调用 trainer 前退出。随后 FIRST_ITER 持续核对 binding 所记
PID=PGID/starttime/cmdline 与 `/proc`，必须看到 `Learning iteration >3500` 且进程仍是同一活进程才写
`phase=first_iter`；自然退出、stale/reused PID 或只有 resume 行都保留 exact identity 人工处置 JSON，不做任何
signal。绝对 checkpoint 用专用
`attest-milestone`，合法值仅 `3700/4000/4500/5500/7500`；source runtime receipt 必须报告 lineage exact=0。

若某行在首迭代前以已审计的 infrastructure-only 原因终止，原行必须先标 `rejected`，保留原目录并绑定
claim/binding/log/launch/identity SHA；不得把旧目录改回 ready。当前仅有两条一次性 `retry_v2`：V1+V2
自由臂的 malloc rc134 行硬绑 GPU1，普通母本保守模仿的 stale-timeout rc125 行硬绑 GPU0。新 claim 必须
绑定 `retry_of`、predecessor 终态证据、`manual_retry_limit=1`、`automatic_retry=false`、
`recipe_equal=true`；loader 会逐字段比较 parent、完整 recipe、seed、budget 和 milestones，并拒绝复用
run name/directory。调度只先选择 GPU1 retry；它的新 claim 可见后才选择 GPU0 retry，所以即使一次请求
`fill --count 9` 也按顺序错峰。每条 retry 在 GPU 锁内、创建新目录前还会重算旧 5/7 个证据文件 SHA，
双次扫描旧 PGID 的 `/proc` 成员，解析 leader/pre-TERM/pre-KILL 成员并逐个要求 absent，再拒绝残留 NVML
context。没有第二次 retry。

旧母本完整 recipe 的边界是原 canonical self-bound queue claim/run binding；v3 会重验其 argv/source/run/lineage，
但不声称用另一份独立真源重新证明历史 recipe。

main 已有独立 per-source+Pod+GPU 的 `1 env × 2 updates` boot-warmup 和 content-bearing 日志默认 180 秒
stale watchdog；P1 不改变其 source-pinned/exact-PGID 语义，也不会让 warmup 继承科学 binding path。
新反例证明 1-env 成功只可当 cache/import probe：同 source/GPU 的正式 4096-env control 仍可能在
`scene_import_start` 后卡死。P1.1 source gate 因此新增上述 full-scene probe，绑定同 source、physical GPU、
task/assets/plant、完整 recipe 与正式 `num_envs`。c7 canary 只闭合旧终档语义；strict caeb probe 已按本页
证据闭合当前 4096-env full-scene 启动/终档门，但尚无科学 trainer 或 Reward 结果。

## Rolling timing 双 Pod 严格续训（2026-07-16）

快速动作、滚动剩余击球时间和预测扰动的 24 格组合不允许走 fresh-only generic runner，也不复用旧的
demo-hotstart snapshot。专用入口先在本机校验冻结 YAML，再对每台 Pod 只建立一次只读 SSH，核验三份唯一 parent：

```bash
python3 scripts/run_phase1_rolling_timing_supercombo_queue.py \
  --queue configs/phase1_rolling_timing_supercombo_20260716.yaml validate
python3 scripts/run_phase1_rolling_timing_supercombo_queue.py \
  --queue configs/phase1_rolling_timing_supercombo_20260716.yaml plan
python3 scripts/run_phase1_rolling_timing_supercombo_queue.py \
  --queue configs/phase1_rolling_timing_supercombo_20260716.yaml inspect-parents
```

`inspect-parents` 完全只读且允许队列尚未激活。它按 YAML 的 exact path/SHA 检查 parent checkpoint、相邻 hard
contract、原始 queue claim、run binding、embedded iteration、全部 floating tensor finite，以及同时存在的
`actor.*`/`critic.*` 权重和非空 optimizer `state/param_groups`；同一 Pod 的多个 parent 在一个连接内顺序检查。timeout 直接返回 UNKNOWN/
失败，不重放、不写远端。

专用 runner 将相对 parent 的 `+200/+500/+1000/+2000` 转成绝对 checkpoint。例如 parent=`model_1600`
对应 `1800/2100/2600/3600`；parent=`model_4700` 对应 `4900/5200/5700/6700`。child claim 同时绑定 parent
checkpoint/hard/原始 claim/binding/RSL directory、专用 runner bytes、完整最终 Hydra argv 与
`formal_evidence_eligible=false`。base recipe 与 job delta 先做确定性 last-write flatten，因此同一个 Hydra key
不能在最终 argv 中出现两次。

RSL-RL 的 `max_iterations` 在 resume 时表示**从 parent 之后再跑多少 update**，不是绝对终点。runner
因此给 trainer 传 `2001`，同时在 plan/claim 中记录 absolute exclusive bound：parent `1600` 的 denominator
是 `3601`、最后预期 checkpoint 是 `3600`，首个 marker 是 `1601/3601`。把 `3601` 直接传给 trainer 会实际
跑到 `5201`；这一真实反例已经进入负测试，
不能再靠字段名猜语义。

激活前先 dry-run；真实发射使用独立确认词：

```bash
python3 scripts/run_phase1_rolling_timing_supercombo_queue.py \
  --queue configs/phase1_rolling_timing_supercombo_20260716.yaml fill --count 24
python3 scripts/run_phase1_rolling_timing_supercombo_queue.py \
  --queue configs/phase1_rolling_timing_supercombo_20260716.yaml \
  fill --count 24 --execute \
  --confirm SIM_ONLY_LAUNCH_ONE_ROLLING_CONTINUATION_JOB
```

runner 按六张卡一圈一条的冻结顺序发射，每张卡上限四条；每条都在创建 run 前原子核验 source/ignored-asset
receipt/Hydra/parent/GPU 容量。父 checkpoint 必须具备 full-state optimizer 恢复资格，child 首个 learning
iteration 必须严格等于 parent iteration `+1`；这两份证据不混写成日志中的虚构 `optimizer=resumed` marker。
失败只保全证据，不自动 retry、迁移或 signal 已有 trainer。

运行态每 30 分钟按队列内 `pruning_contract` 审计。`+200` 只淘汰结构/合同/non-finite/fatal；`+500`
只允许淘汰连续两窗 dense 明显崩坏，不能把缺少 eligible hit 的稀疏零值判失败；`+1000` 仅在同 parent
内按 completion、signed composite、解析回球和 pre/post fall 作容差 Pareto 淘汰，并保留时间覆盖。停臂只按
绑定 numeric PGID 精确处理，不自动 retry。释放的吞吐先加速幸存者；新格只有另有预注册且 source/full-scene
门已过才可占空槽。

## 验证

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_exact_process_group.py \
  hope_training/whole_body_tracking/tests/test_launch_kit_training_locked.py \
  hope_training/whole_body_tracking/tests/test_lean_queue_runtime.py \
  hope_training/whole_body_tracking/tests/test_full_scene_probe_runtime.py \
  tests/test_run_lean_training_queue.py \
  tests/test_run_phase1_rolling_timing_supercombo_queue.py \
  hope_training/whole_body_tracking/tests/test_training_launch_claim.py \
  hope_training/whole_body_tracking/tests/test_training_thread_caps.py
python3 -m py_compile scripts/run_lean_training_queue.py \
  scripts/run_phase1_rolling_timing_supercombo_queue.py \
  hope_training/whole_body_tracking/scripts/exact_process_group.py \
  hope_training/whole_body_tracking/scripts/train.py \
  hope_training/whole_body_tracking/scripts/lean_queue_runtime.py \
  hope_training/whole_body_tracking/scripts/full_scene_probe_runtime.py
bash -n hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh
```

本源码门只证明 YAML 绑定、调度与 fail-closed 选择逻辑；没有证明动作效果或 exam 成绩。整合 focused
source result 为 `126 passed`；现有 source-pinned launcher 同时保留 180 秒 stale-log watchdog 与总 boot
timeout。strict caeb runtime probe 已通过，但它只授权科学队列点火，不能写成科学训练或行为通过。
