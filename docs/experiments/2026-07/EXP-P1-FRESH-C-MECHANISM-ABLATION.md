# Fresh C 单机制首轮消融（2026-07-14）

状态：`Partial`。五个首次发射均为基础设施失败；五个全新 `retry-v2` 命名空间已预注册，尚未启动，
没有训练或行为结论。

## 问题与固定对照

用同一个 fresh C（从零初始化的共享拍面、guidance 权重为零）配方、seed 3、4096 environments、
1001 updates 和 `200/500/1000` checkpoints，只改变一个机制；组合格只改变 V1+V2 两项：

- V1：释放手腕线速度模仿；
- V2：击球窗内动作模仿缩放为 0.25；
- V1+V2：测两者交互；
- 击球前底座减速 reward 权重 1.0；
- 随挥后 replay 起点概率 0.5。

每格都显式写全四轴控制向量，公共 base 不暗含这些轴。关节速度上限 reward 尚无实现，因此第六格
保持 `blocked`。权威机器清单是
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
新 claim。launcher 等到第一个 `Learning iteration`，不是只等早期 Kit marker。`doctor --live` 明确只验
source/assets/module；当前没有可信的 no-Kit Hydra compose 接口，所以不把配置解析冒充已通过。

推荐只用单进程 `fill --count N --execute`：每条依次 doctor、claim、等 first iteration、重采六卡再选
下一条。任何失败立即停止且不自动 retry。所有 `retry-v2` 真正到 checkpoint 前，本实验没有科学结论，
[G05](../../gates/G05_isaac_training_first_loop.md) 保持 `Partial`。

2026-07-14T09:53:56Z 的只读现场验收中，`plan --live` 看到两 Pod 六 GPU 的 compute occupancy 都为 0，
五个旧 claim 均为 Pod1 GPU0 `launched` 且因 queue status=`rejected` 不占新槽；五个 retry-v2 依次分到
Pod1 GPU0/1/2、Pod2 GPU0/1。随后五个 `doctor --live` 全部返回 `DOCTOR_OK`，并实际解析到
`/workspace/hope_isaac_venv/bin/python` 与 exact source module。没有创建 retry-v2 claim 或 trainer；这只
关闭了刚才的 import 基础设施 blocker。
