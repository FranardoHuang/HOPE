# 运行 task-revision exact-0.5 K100

本操作只运行一份 [`0.5 秒时序卷`](../DEFINITIONS.md#timing-exam-0p5)：100 题都从第 0 帧零速度准备态
开始，25 个 50 Hz tick 后触球。[`K100`](../DEFINITIONS.md#q50-and-k100) 的所有未触球、来不及、摔倒和
未上台都保留在分母。结果只是 Isaac inexact 诊断；不能替代 vendor MuJoCo。

当前状态：v1 输入失败已冻结；v2 唯一 Pod2 launch 在首题前暴露 native-clock 安装顺序错误，日志 SHA
`f8c3be8b…a9e28`，没有 scorecard。v2 随后自然结束为 `failed_no_retry`，terminal 文件 SHA
`2d3a9c7d…4894d`、guardian=`D0`，历史 supervisor/evaluator/guardian 与 owned cgroup 均已消失；此前的
exact-stop 尝试在发 signal 前 fail closed，且没有 stop intent/result。禁止重发 v1/v2，也禁止再对旧 PID
执行 stop。v3 唯一 launch 又在任何远端 namespace 创建前被错误的“异常文案只能出现一次”门假拒绝；
日志 SHA 实际完全一致，但 traceback 中源码行和末行各出现一次。v3 已冻结为 `failed_no_retry`。v4 又因
异常类带模块前缀而被冗余的末行文本门假拒绝，且同样零 namespace。当前 v5 彻底删除文本解析，只保留
整文件 SHA+字节数；执行器 SHA `c0fe1555…7c55c`，activation SHA `cb4b8e67…a51886`；执行器、物化器和 timing adapter
合并专项为 `88 passed, 1 skipped`。任何 SSH timeout 都是 `UNKNOWN`，不得重放。

## 1. 本地 source gate（不 SSH、不写远端）

```bash
SOURCE=/path/to/clean/exact-main/nohope
cd "$SOURCE"

test "$(sha256sum scripts/run_phase1_task_revision_0p5_exam.py | awk '{print $1}')" = \
  c0fe155502a1511177fc0702082df37c5e0721ce6629b748bafa48dfe927c55c
test "$(sha256sum configs/phase1_task_revision_0p5_exam_activation_v5_20260718.json | awk '{print $1}')" = \
  cb4b8e6777b5f97a42b60516c3760ba0243cce8ee35d3db4da947db9c2a51886
test "$(sha256sum configs/phase1_task_revision_0p5_exam_v4_failure_20260718.json | awk '{print $1}')" = \
  b7ca44ba34f1ecc396ad3af2984f8c40123f00a12a2d4384bfadc0ce957afed7
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

v5 的 `plan/launch` 会在任何 attempt 写入前、attempt 发布前和资源检查后/创建 cgroup 前三次稳定重读
v2 terminal 与错误日志，并要求三个历史 PID、旧 cgroup 和两个 stop artifact 全部不存在。该门读取的是
自然终档，不会向旧进程发 signal；它还要求 v3/v4 attempt/state/output/cgroup 全部不存在。日志只用
整文件 SHA 与字节数，不再解析 traceback 文本。任何漂移都 fail closed。

## 3. v5 唯一 launch

先生成 v5 plan；它在任何远端消费写入前必须验证上述闭包：

```bash
python3 scripts/run_phase1_task_revision_0p5_exam.py \
  --queue configs/phase1_task_revision_supercombo_20260716.yaml \
  --activation configs/phase1_task_revision_0p5_exam_activation_v5_20260718.json \
  --eval-gpu 1 plan

python3 scripts/run_phase1_task_revision_0p5_exam.py \
  --queue configs/phase1_task_revision_supercombo_20260716.yaml \
  --activation configs/phase1_task_revision_0p5_exam_activation_v5_20260718.json \
  --eval-gpu 1 launch --execute \
  --confirm SIM_ONLY_LAUNCH_ONE_PERSISTENT_TASKREV_0P5_K100_V5
```

v5 使用全新 state/output/attempt namespace；在 gym 创建后先关闭保存的 planner clock owner、安装 native
external command、验证第 0 帧位置不变且速度严格为零，最后才激活 retiming。SSH timeout 后状态是
`UNKNOWN`，**绝不重复 launch**；下一步只能 inspect。
监督器只可清理自己 owned cgroup 内的 evaluator/转换器；不得 signal trainer、worker、其他 evaluator、
真机或宽泛进程名。

## 4. 只读 inspect

```bash
python3 scripts/run_phase1_task_revision_0p5_exam.py \
  --queue configs/phase1_task_revision_supercombo_20260716.yaml \
  --activation configs/phase1_task_revision_0p5_exam_activation_v5_20260718.json \
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
