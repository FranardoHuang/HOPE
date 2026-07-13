# 运行 A0/A1 非击球臂模仿配对

Status: **Partial runtime：A0 正在运行；A1 尚未 claim，只能走 v1r1 continuation**

本页只运行 [A0/A1 非击球臂配对](../DEFINITIONS.md)：A0 保留当前上半身模仿，A1 只从四条
body-imitation Reward 删除左 shoulder/elbow/wrist。它是 simulator-only 单 seed 机制筛查，不启动
judge、不下发真机命令，也不解锁 A2 reward 预算重分配。

> 2026-07-14 现场边界：旧 v1 已启动 A0（PID=PGID `1811464`），随后 outer verifier 假拒绝并退出；
> A1 从未创建。**禁止再次执行 v1 `--mode launch`，禁止重启、停止或改动 A0。** 当前唯一允许的写操作是
> 下文 v1r1 先验证既有 A0 证据、再只创建 A1 的一次性 continuation。

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

### 0.1 安装一次性 control（no-clobber）

从已审阅提交把上述两个文件复制到 Pod。目标根必须不存在；存在就停止审计，不覆盖：

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

### 0.2 plan 与只读 runtime 验证

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

`validate-runtime` 不写文件、不启动进程。它必须同时证明 frozen v1 control、自身 v1r1 字节、训练 source、
A0 三份稳定 SHA和 exact live argv、GPU0 ownership、A1 全面 absent、bank file/metadata physics 绑定，以及
冻结 v1 verifier 仍精确复现原错误。任一失败都不是“可修参数”；停止并保留现场。

### 0.3 只发射 A1

只有审阅上述 JSON 后才执行：

```bash
/workspace/hope_isaac_venv/bin/python \
  /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/run_phase1_non_striking_arm_imitation_a01_v1r1.py \
  --manifest /workspace/codexschema/phase1_non_striking_arm_20260714/control/v1r1/phase1_non_striking_arm_imitation_a01_v1r1_continuation_20260714.json \
  --mode launch-a1 \
  --root-confirm ROOT_APPROVES_SIM_ONLY_A1_V1R1_CONTINUATION
```

runner 先 no-clobber 写 `a0_v1r1_recovery_attestation.json`，再复核 A0/A1 race，之后才创建 A1 claim。
attestation 或 A1 claim 一旦存在，自动重试永久禁止。A1 的 launch contract 同时绑定 frozen v1 与 v1r1
manifest/runner、A0 三份稳定 SHA、recovery attestation 和 exact A1 argv；A1 ready 后还必须证明 A0 未变且
两条 exact trainer 共同拥有 GPU0。代码没有 A0 launch 分支，也不 signal 既有进程。

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
