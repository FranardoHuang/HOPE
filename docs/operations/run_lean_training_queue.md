# 轻量 YAML 训练队列

状态：源码门通过；尚无 Pod 训练结果。G05 仍为 `Partial`。

这个入口解决的是“动作和题库已经决定后，为什么还要手拼一长串命令”。一条 YAML job 必须同时绑定：

- 一个动作名与它的一个或多个 motion 文件；
- 该动作专属的训练 bank 和 immutable exam；
- 一个 clean Git source commit；
- 公共 base recipe 与本臂唯一 delta；
- seed、环境数/迭代预算、`+200/+500/+1000` checkpoint milestone；
- 六 GPU round-robin 资源策略。

它是执行清单，不是第二份优先级账本。job 顺序必须抄自
[`NOW` 统一队列](../NOW.md#统一工作队列唯一优先级账本)中已经解锁的项目；新科学问题仍先写对应实验记录。
示例文件 [`lean_training_queue.example.yaml`](../../configs/lean_training_queue.example.yaml) 故意保持
`blocked`，不会启动 trainer。

## 启动前只查什么

探索训练只做五项快速检查：source checkout 处于 YAML 记录的 commit 且 clean；motion/train bank/exam
三个资产存在；目标 GPU 未达容量；同一个 `run_dir/queue_claim.json` 不存在；Kit 经现有 boot lock 串行启动。
不做逐文件 SHA、`pip freeze`、import closure 或 evidence receipt。那些严格绑定只在正式晋级、跨引擎判卷
和 Gate3 使用。

`setup_train_env.sh`、`scripts/train.py` 与 `scripts/launch_kit_training_locked.sh` 三个入口已固定为 source
checkout 下的 canonical repo-relative 路径，YAML 不能替换成绝对路径、`..`、broad process-control 或
robot runner。`ready` job 还会在任何 SSH 前拒绝全零 commit、非 `/workspace` 路径、placeholder、`..` 和
重复 input identity；blocked 示例可以保留尚未填实的占位值，但永远不会被调度。

Pod 容量固定为 Pod1 每卡最多 4 个本项目 trainer、Pod2 每卡最多 3 个。调度器按
`pod1/gpu0..2 → pod2/gpu0..2` 完成一整圈，才给任一卡放第二条；真实 launch 前用 `nvidia-smi`
把其他人的 compute process 也保守计入占用。并发 `launch-next --execute` 先竞争单控制端全局 scheduler
`flock`；只有持锁者才重新读取两 Pod 六卡、跳过已有 claim、做 round-robin 选槽并启动，因此同一控制端
的多个 agent 不会基于同一份旧快照抢同一槽。远端每 GPU 的 boot lock 仍作最后一道容量/claim 检查。

## 命令

所有模式默认 dry-run，不连 Pod：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue configs/lean_training_queue.example.yaml plan

python3 scripts/run_lean_training_queue.py \
  --queue configs/lean_training_queue.example.yaml status

python3 scripts/run_lean_training_queue.py \
  --queue configs/lean_training_queue.example.yaml launch-next
```

`plan --live` / `status --live` 各 Pod 只建立一个低频只读 SSH，读取实际 GPU 占用。真正启动时，把已过
离线门的 job 改为 `ready`，先看 dry-run 输出，再只启动一条：

```bash
python3 scripts/run_lean_training_queue.py \
  --queue /path/to/active_queue.yaml launch-next \
  --execute --confirm SIM_ONLY_LAUNCH_ONE_LEAN_QUEUE_JOB
```

`launch-next` 每次最多消费一条 `ready` job。`blocked`、`complete`、`rejected` 永远不会进入候选；
预检或 Kit 启动失败后 claim 会保留，禁止自动重试，先诊断再新建明确的后续 job。工具不包含信号、
`pkill`、`killall` 或任何真机入口。

## 验证

```bash
python3 -m pytest -q tests/test_run_lean_training_queue.py
python3 -m py_compile scripts/run_lean_training_queue.py
```

本源码门只证明 YAML 绑定、调度与 fail-closed 选择逻辑；没有证明远端 SSH、Isaac runtime、动作效果或
exam 成绩。
