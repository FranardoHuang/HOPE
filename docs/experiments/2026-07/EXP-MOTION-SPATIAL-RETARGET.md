# EXP-MOTION-SPATIAL-RETARGET — 新动作能否到达有效击球点？

- 状态：in progress（两条只读 runtime inspection 完成；v1 consume activation 已否决；v2 一次性 runner 的源码门与攻击负测通过，但未在 Pod 执行，consume / promotion blocked）
- 阶段/轴：课程阶段 2 / 动作适配与动作源
- 人类负责人：franco
- 执行者：Codex
- 最高证据等级：E2

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

## 2026-07-14：B/C 主选整轨站位已 exact 实体化

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

专项 `10 passed`，全仓 host tests 为 `656 passed, 9 skipped`；两份 exact 私有镜像源先通过只读
`inspect`，随后在 Pod1 的 CPU-only runtime 逐份 `consume` 并以 report-last 原子发布。B/C 最大 root position 逆误差分别
`1.39e-17/2.08e-17`，quaternion 逆误差 `2.22e-16/1.11e-16`，全帧两两距离最大误差
`3.47e-17/4.16e-17 m`，Z bit-exact。B 的 motion/report SHA 为 `27827912...ad6` / `a238c077...df3`，
C 为 `0dd981a6...f48b` / `b3b93d2c...f67`；完整小账见
[`motion_backhand_loop_bc_se2_materialization_results_20260714.json`](../../../configs/motion_backhand_loop_bc_se2_materialization_results_20260714.json)。

这一步只证明冻结主选的整轨刚体搬家逐字节满足合同。它没有 schema-2、L0、vendor L1、桌网整轨、
动力学、simulator、训练或真机证据；证书仍为 `0`，状态继续 promotion blocked，唯一解锁项是另立
schema-2 materialization prereg。materialization 或内部失败必须停止该资产，不能跳 fallback；只有未来桌/网外部
几何失败才回到已冻结 selector 的 `resolve`。

## 2026-07-14：schema-2 前置关节列序合同纠错

在写 B/C schema-2 consumer 前，源码审计确认旧接口文档把两种不同列序错误写成同序：GMR PKL/CSV
`dof_pos` 是腰/头/臂/腿的 controller/MJCF 顺序；Isaac articulation、schema-2 NPZ `joint_pos` 与 ONNX
action 则是从左右髋开始的 interleaved runtime 顺序。现有 `csv_to_npz_mujoco.py` 实际做过重排，
`audit_motion_npz.py` 也把 runtime 顺序另抄了一份，所以问题是合同与重复真源失真，不是已证明的 B/C
输出错序；B/C 还没有 schema-2 输出可追认。

修正后两份明确命名的表
`configs/a3_gmr_dof_pos_joint_order.txt` / `configs/a3_runtime_articulation_joint_order.txt` 由
`configs/a3_joint_order_bijection_v1.json` 绑定原始文件 SHA、canonical names SHA、双向 31 项 permutation、
旧 YAML/Python source-order mirror 与完整 runtime metadata 条件。合同/validator/converter SHA 分别为
`b09987ff...4815` / `8f01d20d...1ae9` / `a151a691...7f04`。source gate 对重复、缺失、额外、长度、
声明 permutation 漂移、duplicate JSON key、wrong-order/partial ONNX metadata 和 NaN/Inf 均 fail closed，
专项 `12 passed`。
MuJoCo converter 已改读同一合同；历史 L0 auditor 因 executed v4 prereg 绑定整份源码 SHA 而保持
byte-exact，新 validator 用 AST 复核其 runtime target literal。没有运行私有资产、forward kinematics、
simulator、RL 或真机。基于 `origin/main@5734dc8` 的 repo-level 回归为
`733 passed, 10 skipped`；缺少可选 runtime 依赖的无边界全目录 pytest 不作为本 source gate 的通过声明。

这个 source gate 仍把 `source_gate_pass_can_authorize_schema2_materialization=false` 写死。B/C 下一步必须
另立 no-clobber prereg，绑定 exact SE(2) PKL、restricted pickle、vendor MJCF/include/mesh、runtime body
order、30→50 Hz、link-origin pose/COM velocity，并明确 B/C root 已在 HOPE frame、不得再旋转一次。当前证书
仍为 `0`，promotion blocked 不变。

## 2026-07-14：B/C schema-2/FK 独立预注册源码门

关节列序纠错后的下一门现已按 B/C 两条独立计划落盘：B prereg SHA 为
`3d71cc02c6ae68d0ecedf280e8341d763ad39ec0aac1757367c9719e761d33ae`，C 为
`662b8c4c0851d2f6d9d5c23313dc0c27334528a2b5fb2b62ad90bc3447257e31`。它们分别绑定 accepted
SE(2) motion/report 的 absolute path、bytes、SHA 和 candidate，不共享输出目录；B/C 从 `91/98`
个 30 Hz frame 按冻结公式得到 `151/163` 个 50 Hz frame。source/internal failure 停止该资产，不能
推进 fallback；只有未来外部桌网失败才回到 frozen selector。

共享 runtime 合同 `configs/motion_backhand_loop_bc_schema2_fk_runtime_v1.json`（SHA
`3d32b146e72029960ebf9cb2777f484804dafc87097e9cd3d0513dc277eed6e8`）绑定：

- accepted restricted NumPy pickle loader 与精确 allowlist/字段集；
- exact formal donor ONNX SHA `0c428ddf...b7b155` 及完整必需子集
  `joint_names/articulation_joint_names/action_joint_ids`；
- vendor MJCF `2ab1cd31...feb97` 与递归 include/external closure：本版 `1` 个 XML、`0` 个
  include、`74` 个唯一 mesh，closure SHA `e0381752...962de`；
- 31 关节 source→runtime permutation、32 个 runtime body column、30→50 Hz linear/SLERP
  和 schema-2 的 link-origin `xpos` / COM `xipos` velocity 分工。

consumer `scripts/materialize_motion_schema2_fk.py`（SHA `33cf23ee...caebd`）的 frame flag 只有
`--hope_frame off` 一个合法值，避免对已经完成 SE(2) 的 HOPE root 二次旋转。两个 tracked `static`
命令均通过；专项 `17 passed`，包含错误 frame/time/point/closure/donor/order、输出重叠、错误 SHA、
duplicate JSON 的 fail-closed 负测；基于 `origin/main@7679b30` 的仓内 `tests/` 回归为
`782 passed, 10 skipped`。此次没有读取
私有 PKL/ONNX，没有导入或运行 MuJoCo FK，没有生成 NPZ，也没有 L0/L1/桌网/动力学/simulator/RL/
真机结果。

donor metadata 文件明确只是与 exact ONNX SHA 绑定的 required-subset 期望，不是已从该 ONNX 抽取的
runtime receipt。因此下一门当时只能先按
[操作文档](../../operations/run_motion_spatial_retarget_screen.md)逐资产执行 no-write `inspect`；下节记录
该门的真实结果，而不是用 source gate 追认 runtime pass。

## 2026-07-14：B/C runtime inspection receipt 与一次性 consume 待审门

Pod1 的独立 detached checkout
`/workspace/codexschema/nohope_schema2_fk_inspect_748b6d5` 保持 exact
`748b6d5fe24bfe58915c34d8dfe09f254f8e4957` 且前后 clean。默认 `/usr/bin/python3`
（Python `3.12.3`、NumPy `2.1.2`）因没有 `onnxruntime` 以 rc=2 fail closed；这次失败没有被算作
inspection pass，也没有写输出。随后只改用现成、未修改的
`/workspace/hope_mjeval_venv/bin/python`：Python `3.12.3`、NumPy `2.5.0`、ONNX Runtime
`1.27.0`、MuJoCo `3.10.0`。同一 tool/plans/donor/PKL/report/MJCF closure 下，B/C 分别返回
`frames=91/98 donor_exact=true no_write=true`，rc 均为 0；两个 schema-2 output root 前后均不存在。

完整历史收据是
`configs/motion_backhand_loop_bc_schema2_fk_runtime_inspection_receipt_20260714.json`（6,689 bytes，
SHA-256 `8e2d2d2d7a4fe0779104456d3bcb32f03cfda82e831958216eefb0fb35b3fb61`）。它绑定 exact checkout、
consumer/plans、donor ONNX、两条 PKL/report、vendor `1 XML + 74 mesh` closure 和解释器/包版本；同时明确
`inspect` 只 restricted-load 输入、重抽 donor metadata 并加载 vendor model/name domain。它没有对
151/163 帧运行 FK、没有写 NPZ、没有推进 dynamics simulation，也没有 L0/L1/训练/真机结论。

最初的 v1 提案 SHA `366d59d5...d6337` 已被红队否决：它允许绕开 activation 直接调用旧
materializer，且失败清理后没有不可逆 claim，所以永远是 **NO-CONSUME** 负结果，不提供运行权限。

替代的 v2 source gate 为：

- activation：`configs/motion_backhand_loop_bc_schema2_fk_consume_activation_20260714.json`，
  SHA-256 `72b22ccd0d691f170d96eba4e0067195a09160c04b12fe9180601115ab546ffb`；
- dependency-light validator：`scripts/validate_motion_schema2_fk_consume_activation.py`，
  SHA-256 `3798122b110571b52909b7f8caedc00dc0898415ffc4653881bcee9dd8b3b536`；
- 一次性 runner：`scripts/run_motion_schema2_fk_consume_once.py`，
  SHA-256 `8e66e0508fec5fc3a973f15fd88c469a6da2ea911e0f3125a1229bdee898a447`。

runner 在 child 前用 atomic hard-link no-replace 发布每资产永久 claim；B/C 共用一个排他
`flock`；同步 `setsid` child 的 stdout/stderr/return code 全量绑定。child 或 post-validation 失败都会
保留 claim 并发布 failure ledger，删除 failure ledger 也不能恢复预算；成功只有在 exact output 内容复核
后才最后发布 success ledger。每次 claim 前重新验证 activation、receipt、runner、validator、detached
clean `748b6d5` checkout、解释器/包/module origins、plans、PKL/report、donor、MJCF closure 和
body/joint order。formal validator 还会打开 NPZ，要求 11 个 exact 字段、schema-2、`151/163 × 31/32`
形状、float32 finite time series、单位四元数以及 exact 32-body order；只有 report hash 而缺这些 lineage
内容不能通过。旧 materializer 的 direct consume 永远不能形成正式结果。

源码攻击负测覆盖 bypass、no-replace、B/C 并发、child failure 后永久花掉 attempt、failure cleanup、
post-validation failure、runtime/module-origin drift、attached/dirty checkout、缺失/伪造 NPZ 与
completion-last lineage；activation/runner 专项 `28 passed`，与原 prereg 合跑 `45 passed`。当前仍是
`review_required_runner_not_executed`：`attempts_started=0`、两个 output root absent、Pod consume 未运行，
L0/L1、桌网、动力学、simulator、训练、正式动作和 hardware 权限全为 false。

权威资料：[G08](../../gates/G08_blind_spot_improvements.md) 和
[操作文档](../../operations/run_motion_spatial_retarget_screen.md)。
