# 运行 task-revision exact-0.5 K100

本操作只运行一份 [`0.5 秒时序卷`](../DEFINITIONS.md#timing-exam-0p5)：100 题都从第 0 帧零速度准备态
开始，25 个 50 Hz tick 后触球。[`K100`](../DEFINITIONS.md#q50-and-k100) 的所有未触球、来不及、摔倒和
未上台都保留在分母。结果只是 Isaac inexact 诊断；不能替代 vendor MuJoCo。

当前状态：持久执行器 SHA `c2ce27845cb26a1ff2474a547556364f72235bcbd830ae5a7768d85fe8141b63`，
activation SHA `996775d6c64a75d4c626d60da20fc52ec27ca86548008aeac900c380de87cfb6`；专项
`35 passed, 1 skipped`。Pod delegated cgroup-v2 probe、launch 与行为结果均未运行，因此保持
`NO_LAUNCH`，直到一次受审操作明确放行。

## 1. 本地 source gate（不 SSH、不写远端）

```bash
SOURCE=/path/to/clean/exact-main/nohope
cd "$SOURCE"

test "$(sha256sum scripts/run_phase1_task_revision_0p5_exam.py | awk '{print $1}')" = \
  c2ce27845cb26a1ff2474a547556364f72235bcbd830ae5a7768d85fe8141b63
test "$(sha256sum configs/phase1_task_revision_0p5_exam_activation_v1_20260717.json | awk '{print $1}')" = \
  996775d6c64a75d4c626d60da20fc52ec27ca86548008aeac900c380de87cfb6
pytest -q tests/test_run_phase1_task_revision_0p5_exam.py
```

本地没有 writable delegated cgroup-v2 child 时，预计为 `35 passed, 1 skipped`；不能把该 skip 写成 Pod
runtime 通过。

## 2. 只生成精确计划

先由只读资源审计选择 `$GPU`；不得碰有 trainer/evaluator context 的卡。2026-07-17 13:05Z 的快照是
Pod2 三卡空闲，但启动前必须重查，旧快照不授权发射。

```bash
GPU=0
python3 scripts/run_phase1_task_revision_0p5_exam.py \
  --queue configs/phase1_task_revision_supercombo_20260716.yaml \
  --activation configs/phase1_task_revision_0p5_exam_activation_v1_20260717.json \
  --eval-gpu "$GPU" plan

python3 scripts/run_phase1_task_revision_0p5_exam.py \
  --queue configs/phase1_task_revision_supercombo_20260716.yaml \
  --activation configs/phase1_task_revision_0p5_exam_activation_v1_20260717.json \
  --eval-gpu "$GPU" launch
```

两条命令都不 SSH。计划必须精确选择 `taskrev_p2_equal_reward@5700`、Pod2
`162.43.172.181:13146`，且写明自动重试、trainer signal、robot command 与 broad signal 全为 false。

## 3. 唯一 launch（当前未执行）

只有负责人确认 source/资源和 Pod delegated cgroup-v2 支持后，才允许**一次**：

```bash
python3 scripts/run_phase1_task_revision_0p5_exam.py \
  --queue configs/phase1_task_revision_supercombo_20260716.yaml \
  --activation configs/phase1_task_revision_0p5_exam_activation_v1_20260717.json \
  --eval-gpu "$GPU" launch --execute \
  --confirm SIM_ONLY_LAUNCH_ONE_PERSISTENT_TASKREV_0P5_K100
```

发射前置会在 commit token 和 evaluator 之前实测 owned cgroup 的进程迁移与 `cgroup.kill`。不支持即
fail closed/`NO_LAUNCH`。SSH timeout 后状态是 `UNKNOWN`，**绝不重复 launch**；下一步只能 inspect。
监督器只可清理自己 owned cgroup 内的 evaluator/转换器；不得 signal trainer、worker、其他 evaluator、
真机或宽泛进程名。

## 4. 只读 inspect

```bash
python3 scripts/run_phase1_task_revision_0p5_exam.py \
  --queue configs/phase1_task_revision_supercombo_20260716.yaml \
  --activation configs/phase1_task_revision_0p5_exam_activation_v1_20260717.json \
  --eval-gpu "$GPU" inspect --execute
```

`running_exact` 只代表身份/guardian/心跳精确；`complete_inexact_isaac_k100` 才允许读取终档。终档还必须
包含 guardian `D0`、cgroup `populated=0`、删除确认、绑定 K100 scorecard 与 `retry_authorized=false`。
`cleanup_unproven_quarantine`、stale heartbeat、缺 ACK、任何 malformed terminal 都保持 fail closed；不删
namespace、不手改 receipt、不重发。

## 5. 结果边界

- 本卷回答 exact 0.5 秒下的实际触球/上台率，不回答动作 TOPP 动力学证书。
- Isaac score 永久 `evaluation_contract_exact=false`；同一 checkpoint 仍须 vendor MuJoCo 同题门。
- 不得据 source gate、`running_exact` 或单侧分数停止 trainer、晋级策略、部署或发真机命令。
- 权威实验记录：[EXP-P1-TASK-REVISION-0P5-K100](../experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md)。

