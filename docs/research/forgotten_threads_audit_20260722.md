# 遗忘线索审计（2026-07-22，起因：Franco 动作从未进训练被集体遗忘）

只读审计全部 NOW 队列、实验登记册、Gate Next Steps 与 preregistered/blocked 记录，
把"承诺过但无人推进"的线索全部列出。本文件是快照存档；行动状态以 NOW 队列为准。

## A. 纯 CPU 就能跑的判卷/评测（最大欠账类）

| 条目 | 出处 | 停在哪一步 | 优先级 |
|---|---|---|---|
| Ready-to-strike v8 CPU-only TOPP execute | NOW:252-255 | v1–v7 全是基建失败，v8 冻结（91 passed）后"远端尚未执行"，7-19 至今没人发射 | P0 |
| 已收口矩阵/push 终档判卷 | 矩阵实验判卷合同 | 终档 model_10700–13900 落盘，全 docs 无一份 judge 结果（2026-07-22 起补：judge_results_20260722.md） | P0（进行中） |
| qdot model_1000 exact 族 immutable judge | NOW:394-396 | 全项目唯一 exact-lineage 正名链入口，未跑 | P0 |
| W/Y/U model_6700 的 0.5 秒 K100 行为卷 | NOW:128 | 卷已双 Pod 物化，demo 双候选从未判 | P0 |
| B 反手拉 vendor 动力学门 + catalog 分级 | 队列[1] | motion_dynamic_replay 正为此合入，B 仍未跑正式分级 | P0 |
| C3/D3 immutable K100 | 队列[2] | 真 blocker=PhysX 关节 8 velocity-limit 刹车与 MuJoCo 不等价的 parity 合同（修复同时解锁 A0/A1、qdot formal 卷）；已补写进 NOW[2] | P0 |
| A0/A1 非击球臂 signed K100 | 非击球臂实验:149-155 | checkpoint 配对闭环后判卷从未激活 | P1 |
| 拍面 p90<15° 毕业量尺 | 课程 v1:16 | evaluator 从未输出该统计 | P1 |

## B. 工具已实现、没人用

- 四个泛化轴工具（TOPP 变速烤入/拍面优化器/引拍缩放/脚步 β）：7-20 合入后零使用，无一份变速评测资产或拍面/引拍评测卷落地。
- 四格机制 canary（热启动/从零 × 引导关/开）：被判卷欠账连锁卡死。

## C. 设计完从未运行 / 无人认领

- READY-SUCCESSOR 孤儿实验（不在 README 索引/NOW 队列；activation 升级草稿在 Franco 本地未提交）。
- T0/T1 连续恢复卷 machine prereg（7-15 起无人做）；原生 MuJoCo 训练四缺口（P0 却零进展）；
  W/Y exact-lineage remediation + Gate3-D0 适配器；S0 高球题族；M0 stance 修复；
  Franco 其余动作段 + v12 挡球管线（六段只消费了 B/C）；teacher first-reset 采用率 probe；
  base-decel 分桶量尺重做；横向躯干扰动实验与 Wave P 的关系未判；稀疏 Reward 资格账本；
  V10 constrained-resume；G07 的 7-02 真机 run 记录（拖 19 天）；G08/球物理老账（P2.1 门、20 次高速击采集）。

## D. 文档矛盾/过期（2026-07-22 已纠偏五处）

README 索引三行状态过期、矩阵实验文件运行表停在"not launched"、NOW[2] 缺真 blocker、
G05 Next Steps 指向已 rejected 的 fresh SZ、DEMO-HOTSTART 状态互相矛盾（待核实）。

## 前 5 个立即行动（CPU 立即可跑 > 解锁量大 > 成本低）

1. Ready-to-strike v8 唯一 execute（Pod2 CPU）。
2. judge.sh 判已收口终档格（进行中，judge_results_20260722.md）。
3. qdot model_1000 两份 checkpoint 的 immutable judge。
4. 修 velocity-limit parity 合同（一次解锁三条 formal 判卷线）。
5. W/Y/U 的 0.5 秒 K100 evaluator。

**制度化**：本审计每周复跑一次；发现的遗忘项必须当场写回 NOW 队列并给出下一份证据定义。
