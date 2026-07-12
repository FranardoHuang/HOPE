# EXP-<ID> — <可证伪问题>

- 状态：`proposed | preregistered | ready | running | completed | invalidated | blocked | superseded`
- 阶段/轴：
- 集成小目标：
- 人类负责人：
- 执行者：
- 复核/决策负责人：
- 最高证据等级：`E0 | E1 | E2 | E3 | E4 | E5`
- 创建日期/最后复核日期：

共享缩写按 [术语与人话对照](../DEFINITIONS.md) 解释；本实验新造的代号仍需在本文写人话。

## 问题与假设

写明一个问题，以及哪种结果会证伪该假设。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 训练/eval/main commit | |
| 动作/action 集 | |
| 观测/action 合同 | |
| Reward | |
| Plant/engine | |
| 训练/考试 bank 或 schedule | |
| Checkpoint/seed | |

## 实验差异

- 对照：
- 改变的变量：
- 其余固定项：
- 决策规则：
- 停止/无效规则：

## 组成与接口

- 正在隔离的组件：
- 集成小目标所需的其他组件：
- 组件间的接口/交接：
- 组件消融后的联合完成规则：

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

使用 all-attempt 分母。没有数据时写 `未测`，绝不跨不兼容协议推断或取平均。
“物理不摔”必须指明使用哪个 physical-fall 字段；如有 tracking guard/reset，另列其次数，
不得用物理不摔率代替连续稳定性。

## 决定

- 决定：`adopt | reject | inconclusive | superseded`
- 理由：
- 是否已纳入当前 setting：`yes | no`
- 局限/下一个 gate：

## 复现与证据

记录命令、operation 文档、机器可读结果，以及必需的 ignored/本地 asset。
