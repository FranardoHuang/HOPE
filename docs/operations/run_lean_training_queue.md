# 轻量 YAML 训练队列

状态：队列 harness 修复后，五个 `retry-v2` 已越过真实 first iteration 并写出 finite `model_100.pt`；
第六个 qdot-limit 格已进入 ready 队列。G05 仍为 `Partial`，这不是行为晋级。

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

探索训练先在本地把 recipe 编译成单义的 Hydra `key=value` 列表：同一个 key 即使分别写成 `key`、`+key`
或 `++key` 也只能出现一次；recipe 不能改写 seed、预算、run name、device、motion/bank binding 或 launch
claim，且拒绝 Hydra flag、删除语法和 `${...}` interpolation。所有 `run_dir` 在整份 YAML 内必须唯一，
不能等于或位于自身 source 内；`ready` job 还不能放进其他 ready source checkout。这些错误都在 SSH 前失败。

远端快速检查为：source checkout 处于 YAML 记录的 commit 且 clean；motion/train bank/exam 三类资产存在；
同一个 child environment 下 `whole_body_tracking` exact 解析到该 source；用最终训练 override 向量实际执行
`train.py --cfg job --resolve`，只做 Hydra 配置合成而不启动 Kit；目标 GPU 未达容量；以一次原子 `mkdir`
创建全新的 run directory；Kit 经现有 boot lock 串行启动。
不做逐文件 SHA、`pip freeze`、import closure 或 evidence receipt。那些严格绑定只在正式晋级、跨引擎判卷
和 Gate3 使用。

`setup_train_env.sh`、`scripts/train.py` 与 `scripts/launch_kit_training_locked.sh` 三个入口已固定为 source
checkout 下的 canonical repo-relative 路径，YAML 不能替换成绝对路径、`..`、broad process-control 或
robot runner。`ready` job 还会在任何 SSH 前拒绝全零 commit、非 `/workspace` 路径、placeholder、`..` 和
重复 input identity；blocked 示例可以保留尚未填实的占位值，但永远不会被调度。run directory 不是
“存在就复用”：任何已有目录、文件或 symlink 都会让原子创建失败，因此旧日志和 state 不会被覆盖。

Pod 容量固定为 Pod1 每卡最多 4 个本项目 trainer、Pod2 每卡最多 3 个。调度器按
`pod1/gpu0..2 → pod2/gpu0..2` 完成一整圈，才给任一卡放第二条；真实 launch 前用 `nvidia-smi`
把其他人的 compute process 也保守计入占用。并发 `launch-next --execute` 先竞争单控制端全局 scheduler
`flock`；只有持锁者才重新读取两 Pod 六卡、跳过已有 claim、做 round-robin 选槽并启动，因此同一控制端
的多个 agent 不会基于同一份旧快照抢同一槽。`nvidia-smi` 偶尔会为同一 trainer PID 返回重复行；live
snapshot 与远端最后容量检查都按每 GPU 的唯一纯数字 PID 计数，重复行不能把一条 trainer 算成两条。
远端每 GPU 的 boot lock 仍作最后一道容量/claim 检查。

`doctor` 与 trainer 共用一个 child-environment builder，同时设置所选 CUDA 和
`PYTHONPATH=${HOPE_WBT_PYTHONPATH}`，并共用同一条最终 training argv。exact module probe 和 no-Kit Hydra
compose 都位于 `mkdir/claim` 之前；失败不会污染新 namespace。claim 的 canonical content 自动绑定 source
commit/checkout、完整 caller argv、run name、seed/预算/milestones、motion/bank/exam identity、Pod/GPU；
其 digest 作为 `++training_launch_claim_sha256=<sha256>` 自动加入 compose 与真实 trainer argv，完整执行 argv
也写入 `queue_claim.json`。这里的 input identity 是路径与语义绑定，不冒充文件内容 SHA。
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
```

`plan --live` / `status --live` 各 Pod 只建立一个低频只读 SSH，读取实际 GPU 占用。`doctor --live`
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

## 验证

```bash
python3 -m pytest -q tests/test_run_lean_training_queue.py
python3 -m py_compile scripts/run_lean_training_queue.py
```

本源码门只证明 YAML 绑定、调度与 fail-closed 选择逻辑；没有证明远端 SSH、Isaac runtime、动作效果或
exam 成绩。当前 boot supervisor 仍只以最终 `Learning iteration` 和固定 900 秒 timeout 作启动裁决；
source-specific asset/cache warmup 的 phase marker 是后续独立改进，不属于本次 P0。
