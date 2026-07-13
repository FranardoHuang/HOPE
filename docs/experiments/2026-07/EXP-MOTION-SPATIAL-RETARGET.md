# EXP-MOTION-SPATIAL-RETARGET — 新动作能否到达有效击球点？

- 状态：completed（proposal screen 与 B/C 确定性主选完成；promotion blocked）
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
plane orientation 前要求有符号 hemisphere 与 B.x `>1e-6`。

首次对真实 v5 输入点火还抓到一个运行前验证器错误：接受结果把 `capture_table_pose_observed=false`
放在 `frame_contract`，而 `frame_contract_evidence` 只保存 path/bytes/SHA；旧读取位置因此在生成任何
proposal 前 fail closed。当前源码改为同时绑定 evidence SHA 和真实 contract 字段；缺失或 true 仍拒绝。

修复后 exact CPU screen 已在 Pod1 对真实 792,241-byte v5 predecessor 完成。640 个动作×题目 cell
中产生 `22` 个 certificate 前 proposal：反手拉 B 为 `19`（覆盖 6 个题格），C 为 `3`（覆盖 2 个题格）；
其余八条为 0。完整外部结果为 225,920 bytes、SHA-256
`69c3db16fa78f526aef49f20eeafe0d7e5e3004c4ed27f5e2823bb3574e2465c`；tracked 摘要
`configs/motion_video_spatial_retarget_signed_results_20260713.json` 的 SHA-256 为
`7fde5725a9950b819c8385e8865cd125d1c629e73efbe7f1aed900069e384d4d`。

本轮结果是 **adopt B/C for certificate work，not adopt as motions**：22 个 proposal 都缺 schema-2、L0、
厂商 L1、整轨桌网余隙和动力学证书，所以 `accepted_candidate_count=0`、`certified_candidate_count=0`，
TOPP/RL/Gate3/真机权限全部为 false。下一步只能物化 B/C 候选并逐张补证；不允许修改 z、尺度、镜像、
关节或逐帧轨迹。其他八条的 0 仍不能跨动作题族判失败，尤其挡球和高点拍压要用自己的题。

## 2026-07-13：B/C 主选与备选顺序冻结

22 个 proposal 不再全部物化。屏后选择合同
`configs/motion_backhand_loop_bc_proposal_selection_prereg_20260713.json`（SHA-256
`691fd516...b9b8c`）绑定上述 225,920-byte 输入和 consumer
`scripts/select_motion_spatial_retarget_candidates.py`（SHA-256 `014db763...d9e23`）。它先按各动作的
名义触球窗筛选 B frame `45–53`、C frame `46–54`，再只把 `yaw=0`、除 `candidate_id/tier` 外逐字段
相同的“仅平移层/偏航加平移层”别名合并，并保留仅平移层。其他任何重复都 fail closed。

每个动作随后按以下键做不舍入的字典序升序：平移范数、偏航绝对值、负回球余量、负身体余隙、frame、
candidate ID。20,084-byte tracked 结果
`configs/motion_backhand_loop_bc_proposal_selection_results_20260713.json`（SHA-256
`8a80a409...8d2be`）选出 exactly one primary per asset：

- B：19 条原始 proposal 中去掉 3 组 `yaw=0` 层级别名后剩 16；主选
  `98e7b883...f3c14`，frame 49，平移 `[0.050360,-0.109156,0] m`、范数 `0.120213 m`、
  yaw `-5°`；冻结 15 条备选。
- C：3 条均唯一；主选 `aa0c86fd...f299`，frame 50，平移
  `[0.157231,-0.157701,0] m`、范数 `0.222691 m`、yaw `-10°`；冻结 2 条备选。

备选不是人工“再挑一次”。只有桌/网外部几何余隙失败可以前进一位；schema-2 物化、L0 静态审计、
厂商 L1 自碰或内部动力学/平衡失败必须停止该动作。consumer 的 `resolve` 模式机械执行这一规则，未知
原因 fail closed。选择 ledger 仍把 materialization/training/TOPP/hardware 全设为 false；下一步必须
另做内容绑定的主选物化与证书合同。host 选择器专项为 `13 passed`，全仓回归为
`646 passed, 9 skipped`。

## 2026-07-14：B/C 主选整轨站位实体化已实现，尚未发布产物

两条主选现在各有独立、不可覆盖（[no-clobber](../../DEFINITIONS.md)）的预注册：

- B `configs/motion_backhand_loop_b_se2_materialization_prereg_20260714.json`，SHA-256
  `e016ca74...51aee`；只消费主选 `98e7b883...f3c14`；
- C `configs/motion_backhand_loop_c_se2_materialization_prereg_20260714.json`，SHA-256
  `27f938cd...9d454`；只消费主选 `aa0c86fd...af299`。

共同 consumer `scripts/materialize_motion_spatial_se2.py`（SHA-256 `21ebbe68...87375`）先绑定
Step-A（前一步确定性主选）结果 `8a80a409...8d2be`，再从 exact counterfactual registry
`fee1b1f9...5529` 逐资产读取 grounded canonical-beta GMR 路径、bytes 与 SHA。它不会从 frozen
ladder 自动取备选。B 对整轨应用平移 `[0.05035998433,-0.109155849041,0] m` 与 yaw `-5°`；C
应用 `[0.157231187588,-0.157700713465,0] m` 与 yaw `-10°`。

这是一个 proper、保地的整轨 [SE(2) 平面刚体变换](../../DEFINITIONS.md)：root position 做
`Rz*p+t`，xyzw root quaternion 做 yaw quaternion 左乘；实际两份源没有 world velocity 字段，若
未来 exact payload 出现显式 `root_{lin,ang}_vel_world`，只旋转而不平移。`dof_pos`、fps、Z、帧数、
`local_body_pos` 和 `link_body_list` 保持 bit-exact；未知 payload 字段、任意 pickle global、非零 Z、
镜像、关节/逐帧/TOPP 编辑均 fail closed。保存重载后再次做逆变换和全帧 root 刚体距离审计，报告最后
发布。

专项 `10 passed`，全仓 host tests 为 `656 passed, 9 skipped`；两份 exact 私有镜像源的只读
`inspect` 也通过：B/C 最大 root position 逆误差分别
`1.39e-17/2.08e-17`，quaternion 逆误差 `2.22e-16/1.11e-16`，全帧两两距离最大误差
`3.47e-17/4.16e-17 m`，Z bit-exact。这里没有执行 `consume`，所以没有 materialized PKL/report、
schema-2、L0、vendor L1、桌网整轨、动力学、simulator、训练或真机证据；证书仍为 `0`，状态继续
promotion blocked。materialization 或内部失败必须停止该资产，不能跳 fallback；只有未来桌/网外部
几何失败才回到已冻结 selector 的 `resolve`。

权威资料：[G08](../../gates/G08_blind_spot_improvements.md) 和
[操作文档](../../operations/run_motion_spatial_retarget_screen.md)。
