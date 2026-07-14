# Fresh C 单机制首轮消融（2026-07-14）

状态：`Partial`。五个首次发射均为基础设施失败；五个全新 `retry-v2` 已越过真实 first-iteration
marker 并持续训练。第六个 qdot-limit tail 机制已完成源码门和 machine prereg，等待同一队列调度；尚无
任何机制的行为晋级结论。

## 问题与固定对照

用同一个 fresh C（从零初始化的共享拍面、guidance 权重为零）配方、seed 3、4096 environments、
1001 updates 和 `200/500/1000` checkpoints，只改变一个机制；组合格只改变 V1+V2 两项：

- V1：释放手腕线速度模仿；
- V2：击球窗内动作模仿缩放为 0.25；
- V1+V2：测两者交互；
- 击球前底座减速 reward 权重 1.0；
- 随挥后 replay 起点概率 0.5。

每格都显式写全四轴控制向量，公共 base 不暗含这些轴。第六格直接惩罚 31 个实际关节速度接近 runtime
limit 的尾部，仍显式写全前四轴控制值。权威机器清单是
[`phase1_fresh_c_mechanism_queue_20260714.yaml`](../../../configs/phase1_fresh_c_mechanism_queue_20260714.yaml)。

## 首次发射的基础设施负结果

五条 attempt-1 依次在 Pod1 GPU0 创建 claim 并启动子进程，但全部在第 0 update、训练 marker 前 rc=1
退出；进程已死，五目录都只有 `queue_claim.json`、`run.log`、`run.log.launch`，没有 model。根因是
`setup_train_env.sh` 只导出 `HOPE_WBT_PYTHONPATH`，旧队列却直接调用 Python，子进程没有
`PYTHONPATH`，最终为 `ModuleNotFoundError: whole_body_tracking`。这不是五个机制的训练失败。

五个旧 job 已标为 `rejected` 并永久保全；不能删除 claim，也不能复用 run directory。唯一允许的
infra-only retry 是完全继承 motion、bank、exam、source、recipe、seed 和预算的五个 `retry-v2` job。

## 修复与继续门

doctor 和 trainer 现在由同一个 child-environment builder 生成 CUDA 与
`PYTHONPATH=${HOPE_WBT_PYTHONPATH}`。source commit/clean、三类资产和
`find_spec('whole_body_tracking')` 的 exact source origin 在创建 run directory/claim 前完成；失败不留
新 claim。launcher 等到第一个 `Learning iteration`，不是只等早期 Kit marker。该次修复时
`doctor --live` 仍只验 source/assets/module；当时没有可信的 no-Kit Hydra compose 接口，因此没有把
配置解析冒充已通过。

推荐只用单进程 `fill --count N --execute`：每条依次 doctor、claim、等 first iteration、重采六卡再选
下一条。任何失败立即停止且不自动 retry。所有 `retry-v2` 真正到 checkpoint 前，本实验没有科学结论，
[G05](../../gates/G05_isaac_training_first_loop.md) 保持 `Partial`。

2026-07-14T09:53:56Z 的只读现场验收中，`plan --live` 看到两 Pod 六 GPU 的 compute occupancy 都为 0，
五个旧 claim 均为 Pod1 GPU0 `launched` 且因 queue status=`rejected` 不占新槽；五个 retry-v2 依次分到
Pod1 GPU0/1/2、Pod2 GPU0/1。随后五个 `doctor --live` 全部返回 `DOCTOR_OK`，并实际解析到
`/workspace/hope_isaac_venv/bin/python` 与 exact source module。没有创建 retry-v2 claim 或 trainer；这只
关闭了刚才的 import 基础设施 blocker。

## Retry-v2 live 与 checkpoint-100 证据

2026-07-14T10:09:52Z，五条 retry-v2 的 exact PID=PGID 均存活，最新 update 为
`160/146/132/117/103`，日志无 NaN、Inf、Traceback、OOM、malloc 或 Killed。Pod1 GPU0/1/2 各一条，
Pod2 GPU0/1 各一条，Pod2 GPU2 空；两 Pod swap=0。五份 `model_100.pt` 后续均验证 filename=embedded
iteration `100`、76 tensor 全 finite、相邻 schema-3 hard contract SHA 匹配、fresh lineage `1`。
这些只证明训练已正常进入和 checkpoint 可审计；`+200` 前不作机制判断。

## 第六格 qdot-limit tail 的冻结配方

qdot 格使用 clean source `a6ccdc7a1c696ff37878039f1e1d83dea28a2bfa`，仍为 fresh seed3、
4096 environments × 1001 updates、guidance `0`、同一 v4rg 动作/bank/exam/零摩擦 plant。四个既有机制轴
固定为 `free_wrist_vel_mimic=false`、`motion_scale_in_window=1.0`、`base_decel_weight=0.0`、
`post_swing_start_prob=0.25`；唯一新增 treatment 是
`joint_velocity_limit_hinge_weight=-5.0`、margin `0.85`。`-5` 下单个关节恰到 limit 的惩罚约
`0.00363`，31 关节全到 limit 约 `0.1125`，小于 14/10/5 主击球 shaping，且只作用于最后 15% 速度
余量。首格不同时扫描 margin，也不买第二 seed。

`+200` 只作方向筛：要求 checkpoint finite/lineage/contract、qdot reward 实际非零且 finite，并比较
末 21 updates 与前 21 updates 的 limit-tail/关节最大速度趋势；同时 completion/composite 相对既有 fresh-C
方向对照不得下降超过 5 个百分点，pre-strike fall 不得恶化超过 2 个百分点。由于旧 control source 没有
qdot contract 字段，这一格只能 screen；若有正向信号，下一空槽必须在同一 `a6ccdc7` source、同 seed 跑
weight `0` 的 exact matched control 后才允许作因果采用判断。

## 发射 harness P0 收紧（尚未产生新 run）

在 qdot 格发射前，队列入口补上四个反复出错的执行合同，但没有连接 Pod 或改动现有五条 trainer：

- recipe 先编译为单义 Hydra key 集；`key/+key/++key` 视为同一个 key，重复 key、harness 自己生成的
  seed/预算/run/motion/bank/device/claim key、Hydra flag、删除和 interpolation 都在 SSH 前拒绝；
- 整份 YAML 的 `run_dir` 全局唯一，ready run 不能放进 ready source checkout；远端只允许原子创建一个
  从未存在的 run directory，不能覆盖旧 log/state；
- standalone doctor 与 launch 内置 doctor 都用真实最终 override 向量运行
  `train.py --cfg job --resolve`，且位于 claim 前；该路径只做 Hydra compose，不启动 Kit；
- canonical claim content 绑定 source、caller argv、run name、seed/预算/milestones、motion/bank/exam identity
  和 Pod/GPU；其 digest 自动成为真实 argv 的 `training_launch_claim_sha256`，claim 同时保存完整执行 argv。

focused 回归为 `19 passed`，并继续覆盖 active queue。此处只关闭发射前 source-contract 缺口：没有新增
claim、checkpoint、机制成绩或 qdot runtime 证据，G05 继续 `Partial`。
