# EXP-TASK-FIRST-N-ACTION-20260727 — 每个动作能否先学会自己的局部 task，再逐轴泛化？

- 状态：`blocked`
- 阶段/轴：ball-free executor 训练 / 任意 N 动作 / 逐动作 task 泛化
- 集成小目标：五个动作各自在安全、可解释的任务域内形成可供 planner 评估的能力面
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（source/host contract；尚无 Pod Isaac smoke 或训练）
- 创建日期/最后复核日期：2026-07-27 / 2026-07-27

共享缩写见[术语与人话对照](../../DEFINITIONS.md)，完整训练语义见
[task-first 任意动作数合同](../../interfaces/task_first_n_action_contract.md)。

## 问题与假设

问题：不用先采 incoming ball（来球）并在线反解 task，而是从每个动作的 reference strike task
开始，依次扩大位置、标量速度、拍面和 base 范围，能否学出稳定且动作专属的泛化域？

假设：若 task 与参考动作在起点自洽、动作身份对 actor 可见、每动作独立用保守置信界晋级，则五个
动作能从中心任务开始逐步扩域；失败动作会停留/回退，不拖着其他动作一起扩大。

以下任一结果证伪首轮假设：

- 中心任务本身长期不能达到预注册成功下界；
- 同动作的 table hit/physical fall 上置信界超过安全阈值；
- 某动作 starvation、借用其他动作分母或换动作后 one-hot/manifest identity 错位；
- 扩一轴时未扩轴的分布也漂移；
- checkpoint 恢复后 curriculum/sampler/receipt 身份丢失；
- 新正手只能靠撞桌或错误的 wrist-COM speed“通过”。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 训练/eval/main commit | 功能分支最终 commit 待整合后填写；未合入 `origin/main` |
| 动作/action 集 | 候选顺序：`bh_loop_c, fh_block_syn, bh_block, s0_highpress, fh_loop_high`；排除旧 `fh_loop` |
| 观测/action 合同 | `task_first_n5` = `hitter_footwork(177) + face/rho(4) + action_one_hot(5)` = 186-D actor；31-D joint target |
| Reward | ball-free；active terms 必须由 [effective Reward receipt](../../DEFINITIONS.md#effective-reward-recipe)钉住 |
| Plant/engine | Isaac A3 + table；正式值以 composed hard contract 为准 |
| 训练/考试 bank 或 schedule | 训练无 ball/question bank；冻结 ball-conditioned heldout 待生成 |
| Checkpoint/seed | fresh seed 对与 budget 待 manifest/Pod smoke 后冻结 |

旧 `fh_loop` 只从本轮 training view 淘汰，历史 bytes/registry identity 保留审计，不做破坏性删除。
未来 93 条动捕原料也不进入本轮；任意 N 接口先用 5 动作闭环验证。

## 每动作任务包络

每个动作固定一个 `station_center_shift_xy_m` 和 reference strike center。随后只按：

```text
position -> scalar speed magnitude -> face cone -> base residual
```

逐轴扩大，档位固定 `0 / 0.25 / 0.5 / 0.75 / 1`。position 同时平移球拍目标和 base task，
base residual 是最后才加入的相对身体容差。任一动作的课程证据不得汇总到另一动作。
level 0 只做中心 warm-up，四个 perturbation 都为零，**不是默认小泛化**；position `0→0.25`
才是第一次扩大任务域。

晋级看成功率 Wilson 下置信界与 unsafe 率 Wilson 上置信界，且要求最少 attempt 与连续 dwell；
回退用更松的独立阈值形成滞回。具体阈值属于 launch manifest，本记录在授权前不猜数。

## 新正手前置 Gate

候选 source 是 `SHADOW_fh_loop_high_yaw152.npz`，source SHA-256：

```text
7d045fcb036ffa668dede4607cfcc82e789a0db7ab86fd8df9dd52cfd5ac4153
```

当前 source-only 诊断为 98 帧、50 Hz；source anchor frame 54 对应 `1.08 s`，source cycle
`1.94 s`，wrist-COM 速度诊断约 `6.54 m/s`。这三项都**不是**正式 post-retime 行为证书：
source anchor 不是 contact authority，wrist-COM 速度也漏掉球拍 offset 的 `omega × r`。

正式进入 manifest 前必须对 upper/full 两件分别验证：

1. action-specific post-retime `t_hit`；
2. `t_cycle`、共同 ready 回位和 recovery；
3. MuJoCo physical `right_racket` site strike speed；
4. 整轨无 robot/racket-table collision、无地面/身体安全失败；
5. reference return Gate；
6. Isaac filtered table-contact smoke。

station 只比较 `[0,0]`、`[-0.05,0]`、`[-0.10,0] m`，负 X 是远离桌；取 upper/full
共同通过中离原站位最近的一档。当前缺 upper/full 正式输出、grounded collocation trace 和 Pod
smoke，所以 `training_authorized=false`。

## 实验差异

- 对照：每动作 level-0 中心 task。
- 改变的变量：同一动作只扩大当前课程轴一档。
- 其余固定项：motion bytes、station center、动作顺序、Reward、plant、PPO、seed、采样器、
  success/unsafe 定义。
- 决策规则：每动作独立通过 Wilson + dwell 门才晋级；坏证据回退一档；stall 按 manifest 明示
  fail 或 freeze。
- 停止/无效规则：identity/SHA/receipt 不符、NaN、counter 不可达、动作 starvation、
  table/fall 失控、reference collision，或任何 ball/planner/retiming/noise 残留即停止。

### Producer 顺序的独立因果对照

“task-first 看起来更好”可能不是因为 producer 顺序本身，而是新流程恰好只给了可行、自洽、低熵
task。必须另做一个同分布对照，不能用 Reward A/B 替代：

1. 用现有 `ball → adapter` 流程生成并冻结一批通过可行性门的 racket tasks，内容绑定完整顺序、
   动作 UID、位置、标量速度、拍面、base 与原 ball provenance；
2. A 臂不再读 ball，直接按冻结顺序 task-first replay；
3. B 臂保留原 ball-first online producer，但限制为同一批 ball/adapter 输入，并逐项证明最终诱导
   的 racket-task 分布与 A 相同；
4. 两臂冻结同 motion、Reward、plant、PPO、seed、预算和 curriculum（首轮都保持中心/同一固定
   task batch，不动态扩域）。

若相同诱导 task 分布下 A/B 表现一致，当前收益应归因于“task 可行、自洽与低熵”，不是
task producer 的先后顺序。若仍有差异，再检查在线 producer 的时序、噪声、cache 和 actor-visible
tuple，而不是先归因算法。这个对照与三项质量权重的
[Reward A/B](EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)必须分开；Reward A/B 两臂也必须使用
同一冻结 task 分布。

## 组成与接口

- 正在隔离的组件：task generator、动态 actor identity、balanced sampler、逐动作 curriculum、
  attempt ledger 与 exact resume。
- 集成小目标所需的其他组件：五动作 training-authorized bank、table contact truth、effective
  Reward receipt、heldout capability paper。
- 组件间的接口/交接：manifest/action UID → trainer hard contract → per-action heldout →
  [capability artifact/selector](../../interfaces/action_capability_selector_contract.md)。
- 组件消融后的联合完成规则：五动作中心全过、至少一轴扩大、无 identity/unsafe/receipt 破坏，
  再购买完整范围与 selector 留出卷。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| host source/contract suite（尚无科学 `run_name`） | prepared | 无 | E1 待整合复跑 | pytest 日志 | 只测 source 合同 |
| Pod1/Pod2 Isaac 两迭代 smoke（尚未命名） | blocked | 无 | 未测 | 未生成 | 2026-07-27 六卡均占用快照；不得写成失败 |
| 五动作 task-first 长训 | blocked | 未冻结 | 未测 | 未生成 | 新正手 `training_authorized=false` |

## 分动作成绩表

| 动作 | 中心 task 成功 | 当前 position 档 | speed 档 | face 档 | base 档 | table hit | physical fall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bh_loop_c` | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| `fh_block_syn` | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| `bh_block` | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| `s0_highpress` | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| `fh_loop_high` | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

全部使用 all-attempt 分母；本表不放 2026-07-27 的 live trainer 快照，因为那些 run 不是本合同。

## 决定

- 决定：`inconclusive`
- 理由：task-first/N-action source contract 方向成立，但新正手资产和 Pod runtime Gate 未闭合，
  尚无训练行为。
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：先完成新正手 upper/full + station 证书，再做 host union 与 Pod 两迭代
  smoke；只有 receipt 全对才启动 fresh 小预算。

## 复现与证据

操作入口：

- [训练操作](../../operations/run_training.md#task-first-prelaunch)
- [动作片到训练绑定](../../operations/run_motion_clip_to_training_binding.md#task-first-addendum)
- [构建与测试](../../operations/build_and_test.md#task-first-and-capability-source-tests)
