# S0/M0 exact post-GVHMR handoff

- 状态：`completed`
- 人类负责人：Franco
- 执行者：Codex
- 证据等级：E2（真实 runtime `inspect/consume` 与不可覆盖 handoff 已完成）
- 创建日期/最后复核日期：2026-07-13 / 2026-07-20

共享缩写见[术语与人话对照](../DEFINITIONS.md)。本记录只回答一个问题：已经完成的
[`S0`](../DEFINITIONS.md)（单条反手高点拍压结构批）和
[`M0`](../DEFINITIONS.md)（四条横移老师结构批）能否在不丢 lineage、不覆盖证据的前提下，成为下一步
canonical-beta materialization 的精确输入。它不回答动作是否安全、能否击球或机器人能否移动。

## 冻结输入

S0 的 exact output 为 `6259d64e...0b22`（389,976 bytes，88 frames）；M0 依次为：

| 候选 | GVHMR output SHA-256 | bytes / frames |
| --- | --- | --- |
| 左移 1 `lateral_step_left_1` | `2873747a...3733` | 463,448 / 105 |
| 左移 2 `lateral_step_left_2` | `e5db30ad...5db2` | 429,208 / 97 |
| 右移 1 `lateral_step_right_1` | `3a8ecc8c...357e` | 363,672 / 82 |
| 右移 2 `lateral_step_right_2` | `45e532c0...c5f8` | 425,304 / 96 |

五条都已经由 exact GVHMR queue 报告 expected-frame 与 required-tensor finite 结构通过。两份新 prereg
不只抄 output SHA：每批还逐字节冻结 execution record、最终 `queue_state.json`、每条 queue binding、
structural audit、源视频 SHA、ready window 与输出 PT。consumer 会复算所有 bytes/SHA，并要求
`prereg -> execution record -> queue state -> binding -> audit -> PT` 双向一致；缺一层或 state 目录出现
额外 binding/audit 都 fail closed。

上游小型总账为 `configs/motion_video_gvhmr_s0_m0_results_20260713.json`，SHA-256
`08b5e8338ac07a20f18034811167c941fed7168703cad4308bc8b2f1e0569726`；两份 prereg 也绑定并逐字段核对这份
summary。4 fps combined-render contact sheet 的人工复核只看到五条均无明显 limb flip/dropout，S0 手臂路径
和 M0 四个方向的迈步-返回可见；这只是 visual diagnostic，不是机器人脚距、动力学或动作 acceptance。

机器真源：

- S0：`configs/motion_post_gvhmr_s0_prereg_20260713.json`，SHA-256
  `c67f9bb4c8e061fc1e5fa74d7bea91f223c7afed5f72ad6dd99a54ac3d79b0ce`；
- M0：`configs/motion_post_gvhmr_m0_prereg_20260713.json`，SHA-256
  `1630cab13cdd3d32d32a22d2e97866f3c92b111e3f6726a601b7d35c59c7fe3e`；
- consumer：`scripts/consume_motion_post_gvhmr_exact.py`，SHA-256
  `b90ca5e04ae8b0edc2b048712138d991d9c68340246238ecd17d28e2c8085d15`。

## 下游硬顺序

handoff 只允许下一步另建 exact、no-clobber 的 canonical-beta materialization。它绑定旧 Franco 同一
performer cohort 的 donor vector `a03f1642...d9cc6`，但明确把“新五条 PT 已写入该 vector”保持为
`not_run`。之后依次是：

1. 新五条 PT 的 canonical-beta materialization，并验证 beta 之外语义 bit-exact；
2. 绑定 clean GMR、entrypoint/loader、SMPL-X、Python 和每条输入/输出的 robot retarget；
3. 绑定 runtime articulation body order 的
   [schema-2](../DEFINITIONS.md) export：位置是 link origin，线速度是 center of mass；
4. L0 finite/limit/endpoint；
5. 厂商 L1、自碰、桌网 swept clearance 与 dynamics。

每一步都要独立 prereg，前一步 certificate 缺失时后一步不能启动。现有 GMR 是 ignored/private runtime；
本分支没有假设本机具备它，也没有运行 GMR。

## 两个语义防作弊门

S0 是第五动作“反手高点向前拍压”。空挥没有球，`observed_ball_contact=null`，击球效果也是 `null`；
它必须后建高球/拍压专用题，禁止借用拉球题纸给出成功或失败结论。

M0 的 terminal ready 不是“脚越近越好”。在 exact robot-coordinate GMR 后，每条候选必须先去除公共
root XY，并把 heading 对齐到初始准备朝向，然后在 `ready_before` 窗口求
`d_xy = right_foot_xy - left_foot_xy` 的稳健中位数，在 `ready_after` 窗口复算同一二维向量。横向站距与
前后脚错位两个分量都必须保留；双脚并拢、更窄站姿或用绝对双足位姿相等替代都不能通过。foot-site
映射和数值容差必须在 GMR result acceptance 前另行预注册；任一缺失都 fail closed。

## 验证

```bash
S0=configs/motion_post_gvhmr_s0_prereg_20260713.json
M0=configs/motion_post_gvhmr_m0_prereg_20260713.json
python3 scripts/consume_motion_post_gvhmr_exact.py \
  --prereg "$S0" --expected-prereg-sha256 "$(sha256sum "$S0" | awk '{print $1}')" static
python3 scripts/consume_motion_post_gvhmr_exact.py \
  --prereg "$M0" --expected-prereg-sha256 "$(sha256sum "$M0" | awk '{print $1}')" static
python3 -m pytest -q tests/test_consume_motion_post_gvhmr_exact.py
```

2026-07-13 host 结果：两份 static contract PASS；专项测试 `8 passed`。随后 exact runtime
`inspect/consume` 已在证据机完成：S0 handoff 为 4,970 bytes、SHA-256
`d57a93e08513c617f4316924e2ef8d9045e26f960c18c284564b7387bd9a1054`；M0 handoff 为 9,242 bytes、
SHA-256 `60c551503571ae522f1396ee2f9e8617aca53dca1f10dd031b7ee27fe9d088ef`。本 handoff 本身关闭的是
lineage，不是 canonical-beta/GMR 结果；下游后来已完成 canonical-beta 与 exact-GMR 诊断，
分别见 [canonical-beta 实验](motion_canonical_beta_s0_m0_20260713.md)与
[exact-GMR 卷宗](motion_exact_gmr_s0_m0_20260713.md)。

## 结论与未宣称事项

采用这两份合同作为 S0/M0 的唯一 post-GVHMR 入口；不采用目录扫描、basename 推断或手工拼 result
manifest。当时本层没有 canonical-beta 新输出或 GMR；当前两层都已有诊断证据，但 schema-2、
L0/L1、动作安全、击球效果、RL、Gate3 与真机仍无结论。完整运行命令见
[操作文档](../operations/run_motion_post_gvhmr_exact.md)。
