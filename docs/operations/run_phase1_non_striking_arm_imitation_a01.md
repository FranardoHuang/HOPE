# 运行 A0/A1 非击球臂模仿配对

Status: **Partial runtime：A0/A1 均已运行；等待 milestone、终档与同卷判读**

本页只运行 [A0/A1 非击球臂配对](../DEFINITIONS.md)：A0 保留当前上半身模仿，A1 只从四条
body-imitation Reward 删除左 shoulder/elbow/wrist。它是 simulator-only 单 seed 机制筛查，不启动
judge、不下发真机命令，也不解锁 A2 reward 预算重分配。

> 2026-07-14 现场边界：旧 v1 启动的 A0（PID=PGID `1811464`）与一次性 v1r1 补发的 A1
> （PID=PGID `1816234`）都已 ready。**禁止再次执行 v1 `--mode launch` 或 v1r1 `--mode launch-a1`，
> 禁止重启、停止或改动两臂。** recovery attestation 与 A1 claim 已存在，任何重复发射都必须 fail closed。

## 冻结字节

- training source：`353a11419ae8589ed4a374ed97169cd7a50d50a3`，tree
  `184fcb296c09988a7d4b2f5b08168f1584b44b9d`；
- manifest：
  [`configs/phase1_non_striking_arm_imitation_a01_prereg_20260714.json`](../../configs/phase1_non_striking_arm_imitation_a01_prereg_20260714.json)，
  SHA-256 `b2462527b6573ce6accaf8e626fe264c3da10e8994dba133d8f0aeaeed870506`；
- runner：
  [`scripts/run_phase1_non_striking_arm_imitation_a01.py`](../../scripts/run_phase1_non_striking_arm_imitation_a01.py)，
  SHA-256 `716279ec68ea1b1e22cc32e634e38cd9e81d4fc969b059d21ec7a1f8e081489f`；
- control commit：`40db3fe5a61d3643cc6a50188a0615666b2d8d91`；
- Pod/runtime：Pod1 GPU0，`/workspace/hope_isaac_venv/bin/python`；GPU2 不属于本实验；
- 两臂都是 fresh seed `17`、4096 env、1001 update、checkpoint 每 100 update，paired 判读只看
  `model_200.pt`、`model_500.pt`、`model_1000.pt`。

## 0. 当前现场与 v1r1 冻结字节

A0 于 `2026-07-13T19:48:35Z` 启动、`19:49:15Z` ready，稳定证据为：

- exact PID=PGID `1811464`；
- launch contract SHA `4c059aa610479a0aea86e437903daaf350f63c1a38f844fa23c517032d418153`；
- launch state SHA `045518bc488bdf5f80cc96a56ed6efa018785283eeb8e7c8f3bff2c27805a342`；
- emitted hard-contract SHA `14ef410be5bdcc341901b3678d5331a59af89382e07939ad2049210bf68c29f1`；
- `model_200.pt` embedded iteration `200`、floating tensor finite、fresh lineage `1`，并绑定上述 hard SHA；
- v1 精确退出行：`[non-striking-arm-a01] FATAL: hard contract train-bank binding changed`；
- A1 arm claim、同名 training run 与 live process 均 absent。

上面最后一条只描述 v1 假拒绝发生时的输入条件。v1r1 已消费该“一次性 absent”条件并成功补发 A1；
当前运行事实为：

- A1 exact PID=PGID `1816234`，Kit ready；
- A1 hard-contract SHA
  `c85b52a28ad64a667a7b522562842466270b3741591f6daf09afc1d0f7c6b146`；
- A1 runtime receipt 的现场摘要为 `runtime_verified.json` SHA `1277cf…77f4`，recovery attestation SHA
  `604288…e9cb`。这里只有现场提供的前后摘要，不把它们当作完整 SHA 或另造 machine receipt；
- A0 exact PID=PGID `1811464` 保持 untouched；judge 未启动。

旧 verifier 错把 bank `meta_json` 中的 `physics_contract_sha256` 要求为 compact hard-contract
`question_bank` 的 direct leaf。v1r1 改为同时验证 compact record 的真实五字段和 bank file SHA，并独立解析
metadata/source-family 的 physics 绑定；不放宽任何训练、checkpoint 或 lineage 合同。

v1r1 冻结字节：

- manifest：
  [`configs/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json`](../../configs/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json)，
  SHA `addcffa1f4dfa41703050a8bc6011a1d94de246d65213bd5a0e6f475897d5fc3`；
- runner：
  [`scripts/run_phase1_non_striking_arm_imitation_a01_v1r1.py`](../../scripts/run_phase1_non_striking_arm_imitation_a01_v1r1.py)，
  SHA `9f98e36063465d36d49eb19e5eb7d55a4f15dc713e739969321f86d8546aecbb`。

这两份 SHA 已被 recovery attestation、A1 launch contract 和 runtime receipt 消费；**不得就地修补 runner
或 manifest**。任何后续 source bug 都必须用新版本、新 SHA 和不改写本次账的迁移说明处理。

### 0.1 安装一次性 control（已完成；禁止重复）

以下命令只保留首次安装的审计记录；目标现已存在，**不得再次执行或覆盖**：

```bash
test ! -e /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1
mkdir /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1

cp --no-clobber <reviewed-source>/configs/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/
cp --no-clobber <reviewed-source>/scripts/run_phase1_non_striking_arm_imitation_a01_v1r1.py \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/

sha256sum \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/run_phase1_non_striking_arm_imitation_a01_v1r1.py
```

### 0.2 plan 与只读 runtime 验证（runtime 已完成）

以下命令只保留调用形状；不得再把已消费的 A1-absent precondition 当作可重放授权：

```bash
/workspace/hope_isaac_venv/bin/python \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/run_phase1_non_striking_arm_imitation_a01_v1r1.py \
  --manifest /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json \
  --mode plan

/workspace/hope_isaac_venv/bin/python \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/run_phase1_non_striking_arm_imitation_a01_v1r1.py \
  --manifest /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json \
  --mode validate-runtime
```

external `--mode plan` 已知有只读路径 bug：它从 external launcher 的 `parents[1]` 推 repo root，因而把旧
相对 manifest 指向 `control/configs/...` 并在读文件前失败。该失败没有写 attestation/claim，也没有启动
进程；exact repo-source plan 已由新旧 runner `30 passed` 覆盖。不要修改冻结 v1r1 来修它，后续新版本
应分开绑定 source root 与 runtime control root。

`validate-runtime` 不走上述 plan 相对路径，现场已经全绿；它不写文件、不启动进程，并同时证明 frozen v1 control、自身 v1r1 字节、训练 source、
A0 三份稳定 SHA和 exact live argv、GPU0 ownership、A1 全面 absent、bank file/metadata physics 绑定，以及
冻结 v1 verifier 仍精确复现原错误。该验证结果随后已被唯一一次 `launch-a1` 消费；不得再次把 absent
条件当作可复用授权。

### 0.3 只发射 A1（已成功；禁止重复）

以下是唯一一次已成功发射的命令记录，**不得再次执行**：

```bash
/workspace/hope_isaac_venv/bin/python \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/run_phase1_non_striking_arm_imitation_a01_v1r1.py \
  --manifest /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json \
  --mode launch-a1 \
  --root-confirm ROOT_APPROVES_SIM_ONLY_A1_V1R1_CONTINUATION
```

runner 已先 no-clobber 写 `a0_v1r1_recovery_attestation.json`，再复核 A0/A1 race，之后创建 A1 claim。
attestation 或 A1 claim 一旦存在，自动重试永久禁止。A1 的 launch contract 同时绑定 frozen v1 与 v1r1
manifest/runner、A0 三份稳定 SHA、recovery attestation 和 exact A1 argv；A1 ready 后还必须证明 A0 未变且
两条 exact trainer 共同拥有 GPU0。现场已满足这些 launch/runtime 门：A1 PID=PGID `1816234` 且 Kit ready，
A0 PID=PGID `1811464` untouched。代码没有 A0 launch 分支，也不 signal 既有进程；这不是行为通过。

### 0.4 v1r1 终档

两臂自然退出后，禁止回用 v1 finalizer；运行：

```bash
/workspace/hope_isaac_venv/bin/python \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/run_phase1_non_striking_arm_imitation_a01_v1r1.py \
  --manifest /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json \
  --mode finalize
```

它复核完整 old+new control/attestation/launch/runtime 链、原错误、两份 hard contract 唯一 mask 差异，以及
`200/500/1000` 的 filename↔embedded iteration、finite、fresh lineage 与 hard SHA；只写一份 no-clobber
paired checkpoint 账，不启动 judge。

实际运行中 A1 自然退出；A0 在写完稳定的 `model_1000.pt` 后卡在 Kit/Python teardown。A0 日志和模型
近三小时不变，正式 failure regex 无命中，terminal checkpoint 的 iteration/finite/lineage/hard binding
先独立通过。操作员又核对 PGID `1811464` 只有同 PID 一个成员、starttime 与 exact run argv 均不变，
才先向该精确 PGID 发 `TERM`；20 秒无响应后向同一 PGID 发 `KILL`。禁止把这个特例泛化成按命令行
匹配、broad signal 或自动 timeout cleanup。终档 paired result 已由上面的 v1r1 finalizer 发布，SHA-256
`30ba716b4e1dc65e0ab20a69cab074e5863a1759d73c33486fe011511247d7d9`；它仍不启动 judge 或授权第二 seed。

## 以下 v1 首次发射流程仅供审计，当前禁止重跑

若 Pod Git 对象库还没有上述 commit，先通过审阅过的 Git fetch/bundle 恢复；不得在归档训练 checkout
`/workspace/codexschema/nohope` 上切分支或改文件。

## 1. 准备独立 source 与 control

以下路径必须首次创建；存在时停止并审计，禁止覆盖：

```bash
git -C /workspace/codexschema/nohope worktree add --detach \
  /workspace/codexschema/nohope_non_striking_arm_353a114 \
  353a11419ae8589ed4a374ed97169cd7a50d50a3

mkdir -p /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1
```

把 manifest/runner 从 exact control commit 逐字节复制到 `control/v1`，然后核对：

```bash
sha256sum \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/phase1_non_striking_arm_imitation_a01_prereg_20260714.json \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/run_phase1_non_striking_arm_imitation_a01.py
```

训练 source 需要 ignored A3 asset 的普通目录副本，不允许 symlink。只在目标完全不存在时，从 clean
`6d93bcb16c422a2f42748c2dc99432559653480b` 归档 checkout 恢复；manifest 会逐文件重算 46-file tree
SHA，并检查目标仍被 Git ignore。通用恢复边界见 [setup_local_sync](setup_local_sync.md)。

## 2. 默认 plan-only

```bash
/workspace/hope_isaac_venv/bin/python \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/run_phase1_non_striking_arm_imitation_a01.py \
  --manifest /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/phase1_non_striking_arm_imitation_a01_prereg_20260714.json \
  --mode plan
```

这条命令不写 run 目录、不启动 Kit。它必须显示 A0/A1 两条 exact argv，除了 `run_name` 和
`++task.rewards.free_non_striking_arm_mimic=false|true` 外完全相同。

没有另开 25-update throwaway pair。mask 已由依赖无关单测验证；真正运行的 smoke 是 locked Kit 到第一条
`Learning iteration`，随后同一两条长臂继续训练，第一份机制 checkpoint 是 `+200`。

## 3. runtime 只读 preflight

在确认 Pod1 GPU0 没有 Yikang/他人任务后运行：

```bash
/workspace/hope_isaac_venv/bin/python \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/run_phase1_non_striking_arm_imitation_a01.py \
  --manifest /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/phase1_non_striking_arm_imitation_a01_prereg_20260714.json \
  --mode validate-runtime
```

它检查 clean source/tree、关键源码 SHA、A3 ignored asset、两条 motion、schema-3 train bank、source-first
Python module、host available RAM、GPU0 compute PID 和显存。此版本要求 GPU0 初始无 compute process、
每臂启动前至少 6000 MiB free、host available RAM 至少 65536 MiB。这个阈值不是显存实测结论；首次
发射仍必须记录两臂各自启动前后的 `nvidia-smi`。

preflight 失败只报告失败，不得为了通过而改 manifest、删旧目录或杀不属于本实验的进程。

## 4. root 显式点火（历史 v1；当前禁止执行）

只有复核 plan/preflight 后才可执行：

```bash
/workspace/hope_isaac_venv/bin/python \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/run_phase1_non_striking_arm_imitation_a01.py \
  --manifest /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/phase1_non_striking_arm_imitation_a01_prereg_20260714.json \
  --mode launch \
  --root-confirm ROOT_APPROVES_SIM_ONLY_A0_A1_V1
```

runner 先让 A0 越过 Kit boot marker，再发 A1；两条随后并发继续。每臂创建独立
`launch_contract.json`、`run.log.launch`、`run.log` 与 `runtime_verified.json`。locked launcher 在 boot
超时只精确管理它刚创建的该臂 PGID。禁止 `pkill`、`killall`、命令行模式信号和真机命令。

如任一臂失败，保留整个 claim/log，记录 exact PID/PGID 和 GPU/RAM；同一路径自动 retry 被禁止。另一臂
已进入训练时不要因配对失败把它 broad-stop，先交由负责人决定是否保留为非配对诊断。

## 5. 运行中检查

只从各臂 `run.log.launch` 读 PID/PGID，并逐次核对：

- exact PID/PGID 成员、GPU0 util/memory、host available RAM/swap；
- 最新 iteration 和 `model_200/500/1000.pt`；
- `NaN/Inf/Traceback/OOM/malloc/bad_alloc/Killed`；
- A1 日志有且只有四条 `left non-striking arm imitation removed` marker，A0 为零条；
- 训练 checkout 始终 clean exact source commit；
- 不启动 judge，不因单条训练曲线停臂或晋级。

## 6. 终档只读验证

两条自然退出后运行：

```bash
/workspace/hope_isaac_venv/bin/python \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/run_phase1_non_striking_arm_imitation_a01.py \
  --manifest /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1/phase1_non_striking_arm_imitation_a01_prereg_20260714.json \
  --mode finalize
```

finalizer 不发 signal、不启动 judge。它要求三份 milestone checkpoint 文件名和 embedded iteration 一致、
浮点 tensor 全 finite、fresh lineage `1`、checkpoint 内 hard-contract SHA 与本臂相邻
`params/training_contract.json` 一致。A0/A1 的 hard SHA 必须不同且各自绑定 post-override 四项 body list；
删除唯一允许不同的 `motion_imitation_body_names` 字段后，两份完整 hard contract 必须逐项相同。
随后才以 no-clobber 方式给每臂写 `checkpoint_result.json`。结果仍标记
`same_immutable_signed_paper_judged=false`、`stop_or_promote_authorized=false`、
`second_seed_authorized=false`。

## 7. 同卷早判

`+200/+500/+1000` 必须使用另行激活的同一 immutable signed paper。当前 runner 不物化 schedule/paper，
也不自动调用 judge。只有 A1 与 A0 都通过 checkpoint binding、同卷方向与安全门后，才允许给胜者与匹配
对照购买第二 seed。A2 固定预算重分配继续另卷 blocked。
