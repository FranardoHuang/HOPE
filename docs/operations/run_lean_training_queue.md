# 轻量 YAML 训练队列

状态：五个机制 `retry-v2` 已越过 `+500`；qdot 的两次不同 physical-GPU 冷启动都曾卡在 A3 URDF
import，不能解释成 reward 失败。P0 harness 已强制 no-Kit Hydra compose、原子 run-dir 与 canonical
claim；P1 source gate 新增显式 opt-in 的 trainer-owned exact log binding、不可覆盖里程碑取证与 phase telemetry。
P1 尚未用 exact source 在 Pod 运行，G05 仍为 `Partial`，这不是行为晋级。

qdot matched-control 的首次冷启动也在 dynamic URDF import/scene creation 前停住；相同 warning 在成功臂
同样存在，不能拿 warning 字面当根因。旧 namespace 已保全并拒绝，只有 unchanged retry-v2 可再试一次。
launcher 现有默认 180 秒“有内容但 size/mtime 均不再变化”watchdog：任一增长重置计时，marker 在边界
poll 上优先；超时先写 sidecar，再只对 launcher 已验证的 exact PGID TERM→5 秒→KILL，返回 rc125。空日志
仍由 900 秒 hard timeout 管，stat 失败 rc126 fail closed。仍缺 scene-created 等细分阶段 receipt；长期应
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
非 `/workspace` 或重复资产。声明了 `preferred_slot` 的 job 只能在该槽探测，reserved Pod 在 SSH 前拒绝。
远端在 fd8 短锁内重查容量，用 plain `mkdir` 和 no-clobber
`full_scene_probe_claim.json` 拒绝复用 attempt；显式使用 source-pinned launcher 的 900 秒 hard timeout、
180 秒 content-stale watchdog 与 `Learning iteration` marker。

launcher 返回只表示 `first_iteration_observed=true`，不表示两次 update 已终档，更不表示 reward 有效。
probe 永远 `not_science=true / attestable=false / promotable=false`，不写 `queue_claim.json`、`run_binding.json`、
`milestones/`，也不能进入 checkpoint attestor、成绩表或自动解锁正式 job。

这个入口解决的是“动作和题库已经决定后，为什么还要手拼一长串命令”。一条 YAML job 必须同时绑定：

- 一个动作名与它的一个或多个 motion 文件；
- 该动作专属的训练 bank 和 immutable exam；
- 一个 clean Git source commit；
- 公共 base recipe 与本臂唯一 delta；
- seed、环境数/迭代预算、`+200/+500/+1000` checkpoint milestone；
- 一个显式 [`dispatch_pods` 可发射 Pod 集合](../DEFINITIONS.md#dispatch-pods)与其 GPU round-robin
  资源策略。

它是执行清单，不是第二份优先级账本。job 顺序必须抄自
[`NOW` 统一队列](../NOW.md#统一工作队列唯一优先级账本)中已经解锁的项目；新科学问题仍先写对应实验记录。
示例文件 [`lean_training_queue.example.yaml`](../../configs/lean_training_queue.example.yaml) 故意保持
`blocked`，不会启动 trainer。

## 启动前只查什么

探索训练先在本地把 recipe 编译成单义的 Hydra `key=value` 列表：同一个 key 即使分别写成 `key`、`+key`
或 `++key` 也只能出现一次；recipe 不能改写 seed、预算、run name、device、motion/bank binding 或 launch
claim，且拒绝 Hydra flag、删除语法和 `${...}` interpolation。所有 `run_dir` 在整份 YAML 内必须唯一，
不能等于或位于自身 source 内；`ready` job 还不能放进其他 ready source checkout。这些错误都在 SSH 前失败。

远端快速检查为：source checkout 处于 YAML 记录的 commit 且 clean；motion/train bank/exam 三类资产存在；
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

Pod 容量上限固定为 Pod1 每卡最多 4 个本项目 trainer、Pod2 每卡最多 3 个；但真正可发射的机器由
`dispatch_pods` 显式收窄。2026-07-14 起 active queue 设为 `[pod2]`，Pod1 全留给 Yikang：调度器仍只读
Pod1 的旧 claim 以防同一 job 在 Pod2 重复发射，却绝不会把新 assignment 放到 Pod1。Pod2 按
`gpu0 → gpu1 → gpu2` 完成一整圈，才给任一卡放第二条；真实 launch 前用 `nvidia-smi`
把其他人的 compute process 也保守计入占用。并发 `launch-next --execute` 先竞争单控制端全局 scheduler
`flock`；只有持锁者才重新读取两 Pod 六卡、跳过已有 claim、做 round-robin 选槽并启动，因此同一控制端
的多个 agent 不会基于同一份旧快照抢同一槽。`nvidia-smi` 偶尔会为同一 trainer PID 返回重复行；live
snapshot 与远端最后容量检查都按每 GPU 的唯一纯数字 PID 计数，重复行不能把一条 trainer 算成两条。
远端每 GPU 的 boot lock 仍作最后一道容量/claim 检查。

旧实现用 `flock FILE command` 包住启动，flock 的私有 fd 会被 detached trainer 继承，导致一条长训把
该 GPU 的 launch lock 持有到终档，配置容量 3 实际退化成 1。现改为短命 controller 显式持有 fd8，并在
调用 launcher 时 `8>&-`；trainer 不再继承，锁只覆盖 doctor→容量→claim→boot marker。现役旧 trainer
已经继承的锁不强行剥离，只等自然退出。job 可选
[`preferred_slot`](../DEFINITIONS.md#preferred-slot)；槽未满时优先，同卡满后自动回到 round-robin，不能突破容量或
`dispatch_pods`。

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
真正批量发射只用一个 `fill` 进程；它持 scheduler lock，逐条 doctor、发射并等到第一个
`Learning iteration`，然后重新读取 claims/GPU 再决定下一条。不要并发调用多个 `launch-next --execute`。

## 命令

所有模式默认 dry-run，不连 Pod：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/lean_training_queue.example.yaml plan

python3 scripts/run_lean_training_queue.py \
  --queue configs/lean_training_queue.example.yaml status

python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml doctor

python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml fill --count 1

python3 scripts/run_lean_training_queue.py \
  --queue configs/lean_training_queue.example.yaml launch-next

python3 scripts/run_lean_training_queue.py \
  --queue configs/lean_training_queue.example.yaml \
  attest-milestone --job-id REPLACE_WITH_JOB_ID --milestone 200
```

`plan --live` / `status --live` 各配置 Pod 只建立一个低频只读 SSH：只把 `dispatch_pods` 的 GPU 计入
可调度占用，但读取全部旧 claim 防重复。`doctor --live`
不创建 run directory、claim 或 Kit 进程；它验证 source/assets/exact module origin，并以真实最终 override
向量执行 `train.py --cfg job --resolve`。只有返回 `hydra=exact-no-kit-compose` 才通过。真正启动前先看
`fill` dry-run，再用单个 scheduler 进程发射指定上限：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml doctor --live

python3 scripts/run_lean_training_queue.py \
  --queue configs/phase1_fresh_c_mechanism_queue_20260714.yaml fill --count 1 \
  --execute --confirm SIM_ONLY_LAUNCH_ONE_LEAN_QUEUE_JOB
```

`fill` 最多消费 `--count` 条，并在每条 first iteration 后重采现场；`launch-next` 仅保留单条诊断用途。
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
`log_dir_bound`、`scene_import_start/done`；冻结 launcher 看到真实 `Learning iteration` 并成功返回后，
queue harness 才在 `.launch` 写 `phase=first_iter`。这能把“卡在 A3 import”与学习失败分开。

main 已有独立 per-source+Pod+GPU 的 `1 env × 2 updates` boot-warmup 和 content-bearing 日志默认 180 秒
stale watchdog；P1 不改变其 source-pinned/exact-PGID 语义，也不会让 warmup 继承科学 binding path。
新反例证明 1-env 成功只可当 cache/import probe：同 source/GPU 的正式 4096-env control 仍可能在
`scene_import_start` 后卡死。P1.1 source gate 因此新增上述 full-scene probe，绑定同 source、physical GPU、
task/assets/plant、完整 recipe 与正式 `num_envs`，但尚未在 Pod 产出真实 probe claim；源码通过不能冒充
full-scene runtime 已通过。

## 验证

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_lean_queue_runtime.py \
  tests/test_run_lean_training_queue.py \
  hope_training/whole_body_tracking/tests/test_training_launch_claim.py \
  hope_training/whole_body_tracking/tests/test_training_thread_caps.py
python3 -m py_compile scripts/run_lean_training_queue.py \
  hope_training/whole_body_tracking/scripts/train.py \
  hope_training/whole_body_tracking/scripts/lean_queue_runtime.py
bash -n hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh
```

本源码门只证明 YAML 绑定、调度与 fail-closed 选择逻辑；没有证明远端 SSH、Isaac runtime、动作效果或
exam 成绩。focused source result 为 `73 passed`；现有 source-pinned launcher 同时保留 180 秒 stale-log
watchdog 与总 boot timeout。P1.1 已提供代表性 full-scene probe 的 source mode，但尚无远端 claim/首迭代
运行证据，仍不能写成 full-scene runtime 通过。
