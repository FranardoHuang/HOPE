# 运行 task-revision exact-0.5 K100

本操作只运行一份 [`0.5 秒时序卷`](../DEFINITIONS.md#timing-exam-0p5)：100 题都从第 0 帧零速度准备态
开始，25 个 50 Hz tick 后触球。[`K100`](../DEFINITIONS.md#q50-and-k100) 的所有未触球、来不及、摔倒和
未上台都保留在分母。结果只是 Isaac inexact 诊断；不能替代 vendor MuJoCo。

当前状态：v1 输入失败已冻结；v2 唯一 Pod2 launch 在首题前暴露 native-clock 安装顺序错误，日志 SHA
`f8c3be8b…a9e28`，没有 scorecard。v2 随后自然结束为 `failed_no_retry`，terminal 文件 SHA
`2d3a9c7d…4894d`、guardian=`D0`，历史 supervisor/evaluator/guardian 与 owned cgroup 均已消失；此前的
exact-stop 尝试在发 signal 前 fail closed，且没有 stop intent/result。禁止重发 v1/v2，也禁止再对旧 PID
执行 stop。当前执行器 SHA `2c117866…445b`，v3 activation SHA `68f2a5eb…404d`；执行器、物化器和
timing adapter 合并专项为 `88 passed, 1 skipped`。任何 SSH timeout 都是 `UNKNOWN`，不得重放。

## 1. 本地 source gate（不 SSH、不写远端）

```bash
SOURCE=/path/to/clean/exact-main/nohope
cd "$SOURCE"

test "$(sha256sum scripts/run_phase1_task_revision_0p5_exam.py | awk '{print $1}')" = \
  2c117866b3fa7a39d5d8480567c0d192668ea068eebd763399f339bf40c8445b
test "$(sha256sum configs/phase1_task_revision_0p5_exam_activation_v3_20260717.json | awk '{print $1}')" = \
  68f2a5eb75bb626194de4ee83da0c01283fd52fb9f385779f317cefda245404d
test "$(sha256sum configs/phase1_task_revision_0p5_exam_v1_failure_20260717.json | awk '{print $1}')" = \
  f53a68136feca34af5e6f7764e11c734f6e151edfff4375ca4a23e29bd611728
pytest -q \
  tests/test_run_phase1_task_revision_0p5_exam.py \
  tests/test_materialize_phase1_timing_exam_0p5.py \
  hope_training/whole_body_tracking/tests/test_isaac_timing_exam_adapter.py
```

本地没有 writable delegated cgroup-v2 child 时，预计合并专项为 `88 passed, 1 skipped`；不能把该 skip 写成 Pod
runtime 通过。

## 2. v2 自然终档前置门

v3 的 `plan/launch` 会在任何 attempt 写入前、attempt 发布前和资源检查后/创建 cgroup 前三次稳定重读
v2 terminal 与错误日志，并要求三个历史 PID、旧 cgroup 和两个 stop artifact 全部不存在。该门读取的是
自然终档，不会向旧进程发 signal；任何 terminal/log/SHA/PID/cgroup 漂移都永久 fail closed。

## 3. v3 唯一 launch

先生成 v3 plan；它在任何远端消费写入前必须验证上述自然终档闭包：

```bash
python3 scripts/run_phase1_task_revision_0p5_exam.py \
  --queue configs/phase1_task_revision_supercombo_20260716.yaml \
  --activation configs/phase1_task_revision_0p5_exam_activation_v3_20260717.json \
  --eval-gpu 1 plan

python3 scripts/run_phase1_task_revision_0p5_exam.py \
  --queue configs/phase1_task_revision_supercombo_20260716.yaml \
  --activation configs/phase1_task_revision_0p5_exam_activation_v3_20260717.json \
  --eval-gpu 1 launch --execute \
  --confirm SIM_ONLY_LAUNCH_ONE_PERSISTENT_TASKREV_0P5_K100
```

v3 使用全新 state/output/attempt namespace；在 gym 创建后先关闭保存的 planner clock owner、安装 native
external command、验证第 0 帧位置不变且速度严格为零，最后才激活 retiming。SSH timeout 后状态是
`UNKNOWN`，**绝不重复 launch**；下一步只能 inspect。
监督器只可清理自己 owned cgroup 内的 evaluator/转换器；不得 signal trainer、worker、其他 evaluator、
真机或宽泛进程名。

## 4. 只读 inspect

```bash
python3 scripts/run_phase1_task_revision_0p5_exam.py \
  --queue configs/phase1_task_revision_supercombo_20260716.yaml \
  --activation configs/phase1_task_revision_0p5_exam_activation_v3_20260717.json \
  --eval-gpu 1 inspect --execute
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
