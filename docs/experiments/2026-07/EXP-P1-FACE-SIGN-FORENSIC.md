# EXP-P1-FACE-SIGN-FORENSIC — 高解析上台率是否隐去了拍面反号？

- 状态：running
- 工作类型：forensic（只做取证复核，不改变训练配方）
- 阶段/轴：共用判分基础 + 课程阶段 1 / 拍面符号
- 人类负责人：franco
- 执行者：Codex
- 工作分支：`Franco_codex/face-sign-forensic`（尚无可引用的 main commit）
- 最高证据等级：E4 diagnostic；修正后卷未跑

共享缩写见 [术语与人话对照](../../DEFINITIONS.md)。

## 问题

现有 fresh `SZ model_2000` K100 报出较高的解析击球/上台率，但正手 actor raw-A
拍面误差接近 180°。要区分两件事：

1. policy 是否真在用球拍反面；
2. 解析回球器是否因 `orient_normal` 把 `n/-n` 当成同一无向平面，从而把错拍面判绿。

## 已有诊断证据

| 格子 | 正手 raw-A 有符号误差 | 人话解释 |
| --- | ---: | --- |
| model-2000 seed1 | `171.10°` | 接近完全反面 |
| model-2000 seed2 | `172.94°` | 接近完全反面 |
| model-2000 seed3 | `173.39°` | 接近完全反面 |
| model-2000 seed4 | `未测` | 没有正手 exact strike，不得补成“第四个 170° 样本” |

现有同一击球状态的 Isaac/MuJoCo 有符号误差约为 `170.72/171.09°`，而解析回球路径
会先对法向做方向归一。这已构成“分数可能对符号失明”的可复现反例，但还不是
修正后的新考卷结果。

## 预注册决定规则

- 为旧 K100 结果生成 content-bound signed-face 诊断表，保留 raw-A 有符号误差。
- 增加 `n` 和 `-n` 必须得到不同判定的负控；对符号无感就 fail-closed。
- 不改训练合同，不因旧高上台率晋级 checkpoint。
- 最终要与同一 K100 的 Isaac/MuJoCo × physical/analytic 2×2 仪器表对账；只修解析 scorer
  不能证明跨引擎 gap 关闭。

## 当前决定

Fresh `SZ model_2000` 的已有成绩仍保留为“解析诊断卡”，但在 signed-face honesty gate
通过前，不得称为 accepted baseline，也不得用来证明 physical return。

证据入口：[Fresh SZ 稳定性实验](EXP-P1-FRESH-SZ-STABILITY.md)、
[G05](../../gates/G05_isaac_training_first_loop.md) 和 [G06](../../gates/G06_isaac_to_mujoco.md)。
