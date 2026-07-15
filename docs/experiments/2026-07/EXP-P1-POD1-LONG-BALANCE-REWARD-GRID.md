# EXP-P1-POD1-LONG-BALANCE-REWARD-GRID — 连续挥拍平衡与击球 Reward 配比

- 状态：`running`
- 阶段/轴：阶段 1，非击球臂、连续挥拍长度、击球位置/速度/拍面 Reward 配比
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：`E3`（12 条科学 trainer 均越过首迭代）
- 创建日期/最后复核日期：2026-07-15

共享术语见[术语与人话对照](../../DEFINITIONS.md)。本卷回答两个直接问题，而不是复制失败 seed：

1. 左臂不再逐帧模仿老师后，是否能主动用于平衡；这种差异是否只有在连续挥更多拍、积累更多站姿偏差后才出现？
2. 同一击球阶段的位置、速度和拍面三个 Reward 会互相争夺学习容量，怎样配比才既能碰到球又能合法回台？

## 冻结设计

机器队列为
[`phase1_pod1_long_balance_reward_grid_20260715.yaml`](../../../configs/phase1_pod1_long_balance_reward_grid_20260715.yaml)。
共 12 条、共同 seed `3`、4096 environments、10001 updates，每 100 保存；正式观察点是
`200/500/1000/2000/3000/6000/10000`。全部沿用 exact source `2c2d70d6...607e`、现役正反手
`v4rg` 动作与 schema-3 signed-face train bank。

第一组是 `2×3`：非击球臂继续模仿/解除模仿 × episode `10/16/24` 秒。现役 `wrap_teleport=false`
会在每条动作结束后保留物理状态并进入下一拍，所以更长 episode 增加的是同一次连续状态里的挥拍次数；不是只把单拍后的
空等延长。每个时长只和同长度配对比较。

第二组固定其余机制，比较六种击球跟踪强度：总预算约 29 的均分、位置主导、速度主导、拍面主导；低跟踪强度
`4/0.5/0.5`；以及保持 `14/10/5` 比例但总强度加倍。前三个主对照总预算相同，可判配比；低强度和双强度只判
“跟踪总量是否压住了真实回球结果”，不能冒充同总预算配比结论。

## 发射与资源纪律

- 只用 Pod1；三张卡按 GPU0→GPU1→GPU2 逐圈发射，最后每卡四条。
- Pod2 九条现役长训不迁移、不重启。
- 同一 setting 不买第二 seed；只有完整单 seed 曲线与 matched control 过门才可另立预注册。
- `200/500/1000` 只判崩溃、non-finite、合同与机制接线；需要真实击球机会才有的稀疏 Reward 样本不足时必须继续。
- `2000/3000` 看中段，`6000/10000` 才形成完整单 seed 结论。Isaac 结果只作训练诊断，最终仍需 vendor MuJoCo 同卷。

Pod1 2026-07-15 15:54 UTC 快照为三卡均 0 compute PID、0 MiB，archive clean exact `6d93bcb...80b`；
独立 source worktree clean exact `2c2d70d...607e`。忽略资产 materialize receipt 为
`d45eb08a...c6f2`，12/12 live doctor 均返回 `SOURCE_ASSET_OK` 与 `DOCTOR_OK`。科学 trainer 尚未发射；
Pod1 4096-env 非科学 full-scene probe 已到首迭代并 rc0；finalizer 因进程退出后 `/proc` starttime 不可读而把
相同 PID/PGID/argv 错判成 identity mismatch，此账本 bug 不改变 scene 启动事实。科学发射前五条已过首迭代；第六条
“24 秒、非击球臂自由” attempt-1（PGID `2152129`）在动态 URDF import 后日志停滞，launcher 只按该 PGID 在
180 秒门收口。原 namespace 永久保留，队尾只有一个逐字同配方 `retry_v2`；其余六个不同问题继续发射。

随后六个不同格中五个已到首迭代；“双倍击球跟踪总强度” attempt-1（PGID `2157392`）也在动态 URDF
import 后停滞并由同一 180 秒 exact-PGID 门收口。它是另一配方的首次基础设施失败，队尾同样只有一个逐字同配方
`retry_v2`。两条 retry 共用 GPU2 剩余两槽；任一再次在同 phase 失败都不再重试。

2026-07-15 16:40 UTC，两条 retry 都拿到 `KIT_BOOT_READY`；Pod1 三卡最终各四条 trainer，利用率
`97%/93%/97%`、显存约 `23.2/23.1/23.1 GiB`，无外部 compute PID。12 条接受臂均为 PID=PGID、
exact source `2c2d70d...607e`、fatal0。冻结快照如下；iteration 只说明已在学习，不是胜负：

| 人话格 | PID=PGID | iteration / latest checkpoint |
| --- | ---: | ---: |
| 10 秒、左臂继续模仿 | 2148472 | 197 / 100 |
| 10 秒、左臂自由 | 2149153 | 194 / 100 |
| 16 秒、左臂继续模仿 | 2149826 | 262 / 200 |
| 16 秒、左臂自由 | 2150538 | 147 / 100 |
| 24 秒、左臂继续模仿 | 2151254 | 142 / 100 |
| 24 秒、左臂自由唯一 retry | 2158534 | 12 / 0 |
| 三项跟踪近似均分 | 2153484 | 68 / 0 |
| 位置主导 | 2154234 | 57 / 0 |
| 速度主导 | 2154953 | 86 / 0 |
| 拍面主导 | 2155850 | 44 / 0 |
| 低强度跟踪 | 2156650 | 38 / 0 |
| 双倍总强度唯一 retry | 2159350 | 6 / 0 |

原失败 `2152129/2157392` 均已确认 non-live、0 iteration、0 checkpoint。下一次只在 model-200/500
做 finite/lineage/contract 与机制接线检查；不因需要击球后才出现的稀疏结果为零而停止。

## 判读边界

现役 `physical_ball=true` 只提供 Phase-A 物理球指标；训练回球结果仍来自达到击球窗后才触发的 VirtualBall 反事实。
因此早期“回台 Reward 为零”可能表示还没有足够挥拍机会，不能直接判 setting 失败。主结果分母必须是所有合法挥拍机会，
不能只除以已经碰到球的样本；机会计数合同进入 main 后，从新 checkpoint 窗口开始使用，不倒灌旧日志。
