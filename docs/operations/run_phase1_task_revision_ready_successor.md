# 运行 task-revision ready successor 四格

本操作只服务于
[准备态量尺 successor 实验](../experiments/2026-07/EXP-P1-TASK-REVISION-READY-SUCCESSOR.md)。它先用一次
Pod2 full-scene probe 证明准备态整数账本可测，再从同一个 `model_5700` full-state parent 启动四格：
baseline/strong ready Reward × [`qdot-limit hinge`](../DEFINITIONS.md#qdot-limit-hinge) `-5/0`。

`qdot-limit hinge` 是超出实际关节速度限位后才收费的惩罚，不是随机横向推力。这个入口不授权随机推力、
Pod1、judge、vendor MuJoCo 或真机。

## 当前状态

**`Partial / NO-LAUNCH`。** 当前实现只暴露 `validate`、`plan`、`inspect-parent`、
`full-scene-probe` 和 `finalize-full-scene-probe` 五个入口。runner SHA 为
`2cf2f3ddecbc7bc1d685254f19e71e59c7dc7be58aae002c74b45f5f4e3b5c8f`，专项 `32 passed`；Pod2 parent
只读语义检查已通过并写回 queue，但尚无 full-scene probe 或 activation evidence。fill、行为/组合判读与
exact stop 都是下一迭代 blocked 接口，当前不存在可执行命令。`launch_authorized: false` 时 science cell
不能启动。

权威输入：

- queue：`configs/phase1_task_revision_ready_successor_20260717.yaml`；
- runner：`scripts/run_phase1_task_revision_ready_successor_queue.py`；
- source commit：`d7c38fcf70e7e9420800437fd5b467168ae72580`；
- Pod2 source root：`/workspace/codexschema/nohope_main_d7c38fcf`；
- Pod2：`root@162.43.172.181:13146`，SSH key `~/.ssh/id_ed25519_runpod`；
- 结果 root：`/workspace/codexschema/phase1_task_revision_ready_successor_20260717`。

所有命令默认 dry-run。带 `--execute` 的命令都是一次性、不可覆盖操作；SSH timeout 只能记为 `UNKNOWN`，
不得以同一 attempt/activation 重发。runner 只允许精确 job/PID/PGID 身份，不得使用 `pkill`、`killall`、
`pgrep -f` 信号或旧 Gate3 broad-kill 脚本。

## 0. 自动任务与本地 source 门

修改 source、queue、runner 或 activation 前，先暂停现有 Pod 自动任务。完成代码、测试、文档并合入 main
后，才可按新的 exact main 状态恢复自动任务；不能让旧自动任务和一次性 mutation 同时管理 Pod。

从 clean checkout 运行：

```bash
QUEUE=configs/phase1_task_revision_ready_successor_20260717.yaml
RUNNER=scripts/run_phase1_task_revision_ready_successor_queue.py

python3 "$RUNNER" --queue "$QUEUE" validate
python3 "$RUNNER" --queue "$QUEUE" plan
```

`validate` 和 `plan` 不连接 Pod。当前 paper 必须显示 `launch_authorized=false`，parent evidence 为 pass、
probe evidence 为 pending；这不是错误，更不能手工跳过。

## 1. 只读绑定 `model_5700` parent

先 dry-run：

```bash
python3 "$RUNNER" --queue "$QUEUE" inspect-parent
```

dry-run 只应列 Pod2 单 SSH 与 read-only candidate。确认后唯一执行：

```bash
python3 "$RUNNER" --queue "$QUEUE" inspect-parent \
  --execute \
  --confirm SIM_ONLY_INSPECT_ONE_READY_SUCCESSOR_PARENT
```

该只读检查已经执行并通过；冻结的 inspection content SHA 为 `e17cedb1…ade4`，evidence content SHA 为
`85967393…1096`。inspector 以 `O_RDONLY|O_NOFOLLOW` stable read 验证原 queue
claim/binding/milestone/checkpoint/hard
contract、known claim content digest、embedded `5700`、full optimizer、schema-3 与所有 tensor finite，
输出每个 path 的 file/content SHA 和机器可写回的 `parent_selection_patch`。它只用 Pod2 一条 SSH，不查
PID/GPU，不写 Pod、不创建 receipt/trainer、不 signal 或 retry。输出是待人工复核的 candidate；runner 不会
自行改 YAML。

## 2. 唯一 full-scene probe

操作者先登记一个全新的 safe-id（只含字母、数字、点、下划线或连字符）；同一个 id 只允许先 dry-run、
再 execute、最后 finalize，失败后不得复用：

```bash
ATTEMPT_ID=ready_ledger_probe_20260717_a1

python3 "$RUNNER" --queue "$QUEUE" full-scene-probe \
  --attempt-id "$ATTEMPT_ID"
```

dry-run 必须显示：Pod2 GPU1、4096 environments、2 additional PPO updates、source
`d7c38fcf…580`、该 exact attempt id 的 fresh no-clobber namespace、自动 retry=false。若 dry-run 显示该
namespace/claim 已存在，停止并登记新的 id；不要删旧证据。唯一 execute 为：

```bash
python3 "$RUNNER" --queue "$QUEUE" full-scene-probe \
  --attempt-id "$ATTEMPT_ID" \
  --execute \
  --confirm SIM_ONLY_RUN_ONE_READY_SUCCESSOR_FULL_SCENE_PROBE
```

不得因为本地等待退出、SSH timeout 或暂时看不到 checkpoint 重发。后续只能对同一 attempt 做 dry-run
finalize，再唯一 finalize：

```bash
python3 "$RUNNER" --queue "$QUEUE" finalize-full-scene-probe \
  --attempt-id "$ATTEMPT_ID"

python3 "$RUNNER" --queue "$QUEUE" finalize-full-scene-probe \
  --attempt-id "$ATTEMPT_ID" \
  --execute \
  --confirm SIM_ONLY_FINALIZE_ONE_READY_SUCCESSOR_FULL_SCENE_PROBE
```

finalizer 必须同时验证 generic finalizer 与 ready 专用量尺，最后以 no-clobber 方式发布：

```text
/workspace/codexschema/phase1_task_revision_ready_successor_20260717/
  runs/_full_scene_probes/ready_baseline_qdot_zero/<source-commit>/
  pod2/gpu1/<attempt_id>/ready_successor_probe_result.json
```

### Probe 通过条件

以下条件必须全部成立：

- `ready_phase_sample_count > 0`；
- planner task-entry count 与 ready phase count 相等；
- ready tilt/base-speed/station-offset eligible count 均与 ready phase count 相等；
- foot-contact 与 foot-slip 各自满足 `eligible + unavailable == ready phase`；
- planner legacy-hold violation 和 ready nonfinite 都是 `0`；
- 旧 ready sums/counts、新 witness 与 task-revision counters 完整且整数守恒；
- model filename iteration=embedded iteration，所有浮点 tensor finite，source/hard/claim/binding/lineage 对拍；
- 日志 fatal=0、自然终止、worker/judge/Kit 残留=0。

generic pass 不能替代 ready 专用 pass。任一项失败都保全 namespace，保持 `NO-LAUNCH`，不重放。

## 3. Finalize 后仍保持 `NO-LAUNCH`

probe receipt 与 parent candidate 通过后，当前 runner **不会**自动修改 YAML 或解锁 science cell；queue
仍必须保持 `NO-LAUNCH`。

人工把 probe receipt 与 `parent_selection_patch` 的 exact file/content SHA 和完整字段写回 YAML activation
evidence，替换其余 `PENDING_*`；随后还要在下一迭代实现 fill/行为 consumer，新 commit、合入 main，再从
clean exact main 重跑：

```bash
python3 "$RUNNER" --queue "$QUEUE" validate
python3 "$RUNNER" --queue "$QUEUE" plan
```

runner 不得在 Pod 上原地改 YAML，也不得在旧 commit 上仅切换 `launch_authorized`。当前 source 没有
`fill` 子命令；不得用旧 task-revision runner 绕过这个缺口。

## 4. 下一迭代 blocked 接口

以下接口是 machine paper 已冻结但 runner **尚未实现**的下一迭代工作；本节故意不给命令或 confirm token：

- 只在 probe+parent evidence 被新 commit 绑定后启用四格 fill；
- 新 run 使用 fresh `O_EXCL` claim/binding，前三格按 Pod2 GPU0/1/2 各一条分散，第四格动态进入最空闲
  GPU 并避让 exact-0.5 K100；
- 单格 behavior inspect/attest、同 parent portfolio inspect/attest，以及同时消费两份 receipt 的 exact stop。

没有这些 reviewed source/test/CLI 之前，禁止直接调用旧 runner、手写 SSH trainer 命令、复制 claim 或猜
confirmation token。Pod1 永不访问。

## 5. 已冻结但尚不可执行的 `+200/+500/+1000` 判读

parent 是 `model_5700`，所以三个绝对 checkpoint 是：

| offset | checkpoint | 用途 |
| ---: | ---: | --- |
| `+200` | `model_5900` | resume/finite/合同和 ready/qdot/task-revision 激活 |
| `+500` | `model_6200` | 两个完整 100-update 窗的明显崩坏、安全、准备/平衡债 |
| `+1000` | `model_6700` | 同 parent 四格的 tolerance-aware Pareto |

未来单格 consumer 必须绑定 YAML 四个 job id、绝对 checkpoint `5900/6200/6700`、claim/binding 与两个
互不重叠 100-update 整数窗。当前没有 behavior/portfolio 子命令，因此不能生成或消费下列 planned path。

每个 run 的原通用行为 receipt 保持：

```text
<run_dir>/behavior_milestones/model_<N>.json
```

ready wrapper 另发布：

```text
<run_dir>/ready_behavior_milestones/model_<N>.json
```

未来同 offset 四格到齐后，组合结果计划写到：

```text
/workspace/codexschema/phase1_task_revision_ready_successor_20260717/
  portfolio_decisions/pod2_equal_reward_model5700/offset_<N>.json
```

稀疏合法回台只有在 eligible 分母为正时才可解释；eligible 不足时零值永不淘汰。`+1000` 至少保留两格，
并同时保留 qdot=`-5` 与 qdot=`0` 的覆盖。单格 behavior receipt 和同 parent portfolio receipt 两者都授权
后，未来 exact-stop 才可能进入 dry-run；signal 前仍必须重验 exact numeric process identity。当前没有
该入口，禁止运行 stop。

## 6. 失败处理与证据边界

- 任何失败先保留 stdout/stderr、attempt namespace、claim/binding/checkpoint 与 receipt presence；不自动重试。
- SSH timeout 只标记 `UNKNOWN`；下一轮先只读 inspect，不能把未知当作“没启动”而重发。
- importer/boot/runtime 失败只作基础设施拒绝，不能冒充 Reward 失败。
- source/full-scene/Isaac 结果都不是 vendor MuJoCo 或真机证据；本 operation 禁止任何真机命令。
- 在 probe 与 activation commit 真正完成前，四格始终 `Partial / NO-LAUNCH`。
