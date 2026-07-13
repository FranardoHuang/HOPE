# EXP-MOTION-SPATIAL-RETARGET — 新动作能否到达有效击球点？

- 状态：preregistered
- 阶段/轴：课程阶段 2 / 动作适配与动作源
- 人类负责人：franco
- 执行者：Codex
- 最高证据等级：E1

共享缩写见 [术语与人话对照](../../DEFINITIONS.md)。

十段 Franco/v6/v7 空挥已通过内容寻址入库、canonical GMR/grounding，以及稠密采样的
地面/自碰/身体余隙检查。冻结的 64 题筛选得到零公共支持集，因此无法在两动作与四动作之间做出选择，
也不能证明这些动作无用。

正式有效性改为三者交集：每个动作自己的安全触球时空流形、适配的来球/动作题族，以及合法的
整轨 `SE(2)` 站位。反手拉 B（frame 49，intrinsic `32/32`，最近旧题 `0.165 m`）和 C
（frame 50，`27/32`，`0.237 m`）只满足 `0.30 m` 平移范数粗界，仍是候选；正手先过
signed-face 诚实门，挡球必须另出挡球题。

2026-07-13 源码审计发现原 `d8c918ac...5a9f` prereg 绑定的旧 analytic scorer 会在
`orient_normal` 中抹掉 `n/-n`；该 prereg 尚未执行，现已撤销而不追认。替代 manifest SHA 为
`0f757c8c...af66a`，绑定 scorer `9d01da15...0f5ec` 与 proposal tool
`d053dd50...5259b`：每题按 forehand/backhand `[+1,-1]` 把 raw-A 映射到 physical-B，并在
plane orientation 前要求有符号 hemisphere 与 B.x `>1e-6`。这只是 E1 量尺修正，没有重跑
proposal，也没有改变任何动作的候选/晋级状态。

首次对真实 v5 输入点火还抓到一个运行前验证器错误：接受结果把 `capture_table_pose_observed=false`
放在 `frame_contract`，而 `frame_contract_evidence` 只保存 path/bytes/SHA；旧读取位置因此在生成任何
proposal 前 fail closed。当前源码改为同时绑定 evidence SHA 和真实 contract 字段；缺失或 true 仍拒绝。

下一步只允许在冻结边界内，对整段动作做一次保持接地的 SE(2) 变换；不允许修改 z、尺度、镜像、关节或逐帧轨迹。
目前没有候选 certificate，也没有动作被推广。进入 TOPP/RL/Gate3 前仍需通过 exact schema-2、L0、厂商 L1、
整轨迹桌网余隙和动力学检查；这些门只授予训练资格，不等于回球有效。

权威资料：[G08](../../gates/G08_blind_spot_improvements.md) 和
[操作文档](../../operations/run_motion_spatial_retarget_screen.md)。
