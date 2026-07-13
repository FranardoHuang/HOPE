# 简短进度记录

本文件只保留短日期摘要，不再做第三份实验真源。更新时只写几句话并链接到权威位置：

- 当前 setting/采用状态：[NOW](NOW.md)
- 实验设计与证据：[experiments/](experiments/README.md)
- `main` 上的重要变化：[TIMELINE](TIMELINE.md)
- 可复现验收：[gates/](gates/)
- 缩写与人话释义：[DEFINITIONS](DEFINITIONS.md)

旧 1700 行记录完整保存在
[历史 PROGRESS](experiments/archive/PROGRESS_legacy_through_2026-07-12.md)。

## 2026-07-13

- signed-face v5 在 scene 构建后、第一次学习前被旧 schema-3 train-bank physics contract 正确拒绝；
  A claim/log 保留，B/C/D 未创建，没有 checkpoint。新增严格 no-clobber 重绑定 consumer：只允许一个
  冻结 helper 的加法式源码变化，要求所有问题数组 raw bytes 不变、metadata 精确四 leaf，并在目标
  runtime 重跑 exact motion contract 与 1481 题 old/new bitwise physics replay。v1 no-write Pod
  preflight 又抓到 Python 小版本相关的 `ast.dump` SHA 假拒绝；v2 改用 helper 原始源码片段 SHA、仍
  保留同 runtime AST 等价门。v2 已发布 bank/report SHA `3a9d8851...5b71` / `9fffed03...bb37`，24 数组
  未变，两侧 landing/net 全过；v6 launcher 绑定完整 report 及父旧 bank→当前新 bank 的唯一精确
  common-field transition。专项 `32 passed`；v6 L1 尚未启动，旧 exam family 也未重绑定，故 L2/judge
  继续阻断。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- signed-face 单-seed 漏斗的首次 Pod v1 preflight 在创建 run 前抓到 checkpoint 审计假拒绝：旧代码
  只看顶层 tensor/合同键，实际 RSL-RL 权重嵌套且 provenance 在 `infos`。父 `model_13800.pt` 递归
  `74` 个浮点 tensor、`1,762,715` 元素、nonfinite `0`；v2 改为递归扫描并绑定 `infos`，保留 v1
  证据且不改四格/seed/预算/L2 blocker。v2 随后在首格学习前因 exact worktree `PYTHONPATH` 未传给
  child 退出；失败 claim/log 保留，其他三格未创建。v3 绑定 tracked setup、拒绝 local override，并
  在 claim 前解析模块来源。v3 因在 `SimulationApp` 前真正 import IsaacLab 而假拒绝；v4 改用
  `find_spec` 只验 exact module origin。v4 再在 scene 构建时发现 ignored A3 资产缺失；失败 claim
  保留，v5 从 clean `6d93bcb` 恢复并绑定 source/target `46` files、`15,378,264` bytes、tree SHA
  `0137f59b...26c6`。专项 `23 passed`；v5 Pod launch 尚未记为完成。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- 有符号拍面修复后的首轮消融已从 E0 设计升级为 machine prereg：同卡只跑 seed3 的
  hot/fresh × face-guidance-off/on 四个因果格；热启动明确保持 lineage0，fresh 必须 lineage1，半写
  claim/no-clobber/缺失 Git checkout 均 fail closed。focused `23 passed`；L1 尚无 Pod 行为结果，L2
  在 signed directional checkpoint paper 的 path/SHA 冻结前硬阻断，也没有 judge/真机授权。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- S0/M0 exact post-GVHMR handoff 已在证据机完成，分别为 4,970/9,242 bytes、SHA-256
  `d57a93e0...a1054` / `60c55150...088ef`。下一层 canonical-beta 已做成两份独立 no-clobber prereg：
  复用旧 materializer 的 PT/save-reload 审计，只注入旧 Franco exact donor，不重算新 cohort。host static
  与新旧专项为 `15 passed, 1 skipped`，最新 main 重放回归 `620 passed, 9 skipped`；真实 PT 尚未 consume，
  GMR/schema-2/安全/效果/训练仍未授权。
  M0 的 foot sites、初末二维脚距、容差和 pass 全保持 null，必须由未来 exact GMR 产生。详见
  [canonical-beta 卷宗](experiments/motion_canonical_beta_s0_m0_20260713.md)。

- S0/M0 的五条 exact GVHMR 结果已增加 post-GVHMR no-clobber consumer：两份 prereg 同时绑定 tracked
  summary、execution record、queue state、每条 binding/audit/PT 和 canonical-beta donor，host static
  两批通过，专项 `8 passed`；后续 runtime handoff 已按上一条完成。canonical-beta/GMR/schema-2 仍未运行；S0
  禁止借用拉球题，M0 后续必须恢复含前后错位的初始二维脚间向量，双脚并拢不算成功。详见
  [实验卷宗](experiments/motion_post_gvhmr_s0_m0_handoff_20260713.md)与
  [操作文档](operations/run_motion_post_gvhmr_exact.md)。

- Franco 动作主线第一次从“排队”进入 runtime：Pod1 上 S0 高点拍压 `88/88` 帧、M0 四条横移
  `105/105、97/97、82/82、96/96` 帧全部通过 exact GVHMR finite structural audit；输入、execution
  record、queue、output、binding 和 audit SHA 已进入
  [`motion_video_gvhmr_s0_m0_results_20260713.json`](../configs/motion_video_gvhmr_s0_m0_results_20260713.json)。
  同时 signed spatial-retarget 对真实 v5 输入完成 640-cell screen，反手拉 B/C 分别产生 `19/3` 个
  bounded proposal，但 certificate 仍是 `0`，所以只解锁物化/安全门，不解锁 TOPP、RL、Gate3 或真机。
  详见[GVHMR 小批](experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md)和
  [空间重定位](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)。

- 反手拉 B/C 的 signed spatial-retarget 首次对真实 v5 输入点火，在生成 proposal 前抓到验证器
  schema 假拒绝：`capture_table_pose_observed=false` 位于 `frame_contract`，而旧代码误从只含
  path/bytes/SHA 的 `frame_contract_evidence` 读取。修复后仍同时绑定 evidence SHA，且缺失/true
  fail closed；新 prereg/tool SHA 为 `0f757c8c...af66a` / `d053dd50...5259b`。这只解除输入验证阻塞，
  尚不是动作晋级。详见[动作空间重定位实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)。

- 复盘 Phase-1 的 GPU 证据购买方式：`SZ` 在 2k 已失去稳定性资格后，把四 seed 都继续买到 4k
  对拒绝 baseline 属于过量复现。新制度改为一个阻断 seed 先跑四个不同机制单元，固定
  相对 `+200/+500/+1000` checkpoint；只有胜者和匹配对照补第二 seed，`3–4` seed/terminal 只给正式候选。
  第一张新纸是“热启动/从零 × 线性拍面引导关/开”的四格；当时只有 E0 设计，尚未启动 Pod、训练、
  judge 或真机。详见[机制漏斗](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)和
  [算力制度](research/phase1_ablation_acceleration_2026-07-11.md#seed-是晋级税不是首轮并发单位)。

- signed-face 诚实门已在 feature source 闭合：Isaac virtual reward 与 NumPy/MuJoCo analytic
  scorer 都在 `orient_normal` 前绑定 raw-A、每 clip `[+1,-1]` physical-B 和严格 +X/hemisphere
  门；`n/-n` 负控证明同一冲量/落点下错面不再记分，旧 unsigned 路径只能显式 inexact。seed3
  TensorBoard 九个 milestone 的 content-bound 摘要又显示正手误差 `174.02°`/normal pass `0` 时
  训练回台仍 `.965`；实际 `env.yaml` 绑定启用的 face-blind reward 及 `20/30/5/5` 权重。step13800
  的 `2.961×` 只是跨环境/正反手的全局 reward-tag 比值，不能量化正手错面支付份额；准确结论收紧为
  “wrong-face FH states were treated as reward-eligible by the active face-blind reward path”，而非
  “已量化错面支付”或单因素因果。focused 为 `38 passed, 1 skipped`，
  顶层 broad 为 `546 passed, 9 skipped`；
  没有 simulator/Pod/真机行为结果，fresh canary 与同卷复判仍待做，G05/G06 保持 `Partial`。详见
  [拍面符号卷宗](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。
- Fresh `SZ model_4000` 四 seed 同一 K100 已通过 Linux fake-runner 冒烟、两 Pod 一次性持久
  启动和正式 aggregate 完成：`50/88/98/0`，median `.69`、worst `.00`、spread `.98`、
  worst-side `.00`，四项稳定门全失败；seed4 有 21 次 root fall，判为持续弱而非晚熟。
  seed2/3 正手 parsed `38/50,48/50` 但 signed composite 均 `0/50`，法向误差
  `172.33°/174.35°`，所以旧高分不晋级。aggregate file/content SHA 为
  `1ba88e39...d195` / `226e6050...648d`；详见
  [稳定性实验](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。
- 动作离线顺序纠正为 Franco 主线优先：六段旧素材复用 exact GVHMR/GMR，不再重跑；反手拉 B/C 的
  frame 49/50 只登记为空挥名义视觉锚点。新视频拆成互不阻塞的 [S0/M0](DEFINITIONS.md) 离线结构批：高点拍压单条与四条横移候选，
  v12 本轮不授权。两批绑定 GVHMR/权重/Python/`nvidia-smi`/validator/argv，并用 batch-only source
  fd 生成私有只读快照供 child 消费，再以 inode/mtime/ctime/SHA 复核；同时拒绝 symlink、原子 claim
  不相交 state/output。07-11 旧 launcher 已压成仅证据 gzip，不再提供通用入口；M0 未来终点必须恢复初始、
  朝向对齐且含前后错位的双脚分离向量。Host 聚焦套件 `50 passed`，仓库 `tests/` 为
  `573 passed, 9 skipped`；本分支未复制 Pod、未启动
  GVHMR/GMR/simulator/RL/真机。见
  [实验卷宗](experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md)与
  [操作文档](operations/run_motion_video_gvhmr_prereg.md)。
- Phase-1 fresh 广度池分两波完成负责人批准的运营收口：16 臂已全部在保留日志并验证最后
  checkpoint 的迭代、`1,762,715` 个浮点元素 finite、schema-3、fresh lineage 与相邻合同 SHA 后，
  只按各自登记 PGID 停止。第二波前又确认 24/24 最近 K20 格的正手 signed composite 都为 0；
  TERM 未退出时仅在确认无 live child/Kit-lock holder 后对同一
  exact PGID 使用 KILL，没有 broad kill、worker/judge 信号或真机命令。这不是预注册 q10/q50
  阈值停止结论，旧 `screen_only`/`whole_arm_stop_allowed=false` 语义不变；完整曲线、PGID 和 checkpoint
  SHA 见[拍面×plant 广度实验](experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)。
- exact planner-policy tuple 源码已在 latest-main 集成候选中闭合：23 项有效源码/配置逐字节匹配
  `c0a8e46`，portable Release 为 focused `40/40`、native `233 passed + 5 optional skips`，主线本地
  回归为 planner `180 passed, 2 skipped`、serve `39 passed`、root `521 passed, 9 skipped`。这只关闭
  source/binary merge blocker；ROS/Jazzy/AimRT、formal ONNX runtime、backend first tick、vendor
  MuJoCo 和真机都未运行。详见
  [实验卷宗](experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md)。
- 最新 main 登记并在本地逐字节核验了 7 段私有新视频：v12 正反手挡球、高点拍压第五动作，以及
  左右横移各两段下肢老师。新版 intake 合同能区分挥拍与横移动作，拒绝重复 JSON 键、非有限数和
  角色/动作错配；7/7 文件与 11 项专项测试通过，仓库测试为 `472 passed, 9 skipped`。同时建立了
  [动作组合设计](experiments/motion_v12_high_press_lateral_teacher_20260713.md)、
  [视频 intake 记录](experiments/motion_video_intake_v12_static_motion_20260713.md)和
  [非击球臂消融](experiments/non_striking_arm_imitation_ablation_20260713.md)。这只证明素材登记和
  设计已落地；没有复制到 Pod，也没有 GVHMR/GMR、仿真、RL 或真机行为结果。
- 按负责人阅读路径重写 [NOW](NOW.md)：先解释题目、参考动作、179 维输入、31 维关节目标、
  Reward、PPO 和独立判卷怎样组成一套完整训练，再按现行课程逐段写问题、解法、效果与差距。
  同时纠正阶段编号：阶段 2 是虚拟球变到达状态，站位/脚步是其中的解法；阶段 3 才是物理球
  进场；连续恢复和 `Gate3/Gate3B` 分别是横向能力线和部署验证线。成绩卡明确为 Python
  BankExam 单拍解析诊断，不是 Gate3。本次只改文档，没有新增训练、仿真行为或真机结果。
- Fresh `SZ model_4000` 四 seed 同卷的 Pod1/Pod2 readiness audit 与 all-four activation
  已物化（activation file `9dea76c2...ce704`，content `eaa92ca2...aa4fb`），两 Pod
  `contract-check` 通过。随后两份 no-clobber runtime contract 已完成 `prepare`；Pod1/Pod2
  file SHA 分别为 `2b76a5a...8201e`、`dbecc102...d1c9b`。当前仍是
  `prepared_not_started/jobs_started=0/auto_start=false`，没有 run、judge、新分或真机动作；该
  readiness/prepare 事务当时未发 trainer signal，后续 8 臂运营停止是本节首条记录的独立决定。
  持久监督器 source gate 后续已审绿，仍缺 Linux fake-runner smoke 与正式 job。详见
  [Fresh SZ 稳定性实验](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。
- `model_4000` 同卷启动新增一次性、无覆盖的持久监督器：父进程只在核对 PID=PGID、procfs 身份、固定环境和完整 SHA 闭包后发布不可逆 token；token 可见后的超时、证据 `stat` 或临时清理异常都只能报告 committed-pending，不能产生重试权限。supervisor+queue+consumer 为 `64 passed`；这仍是 host 源码门，Linux/Pod 与 MuJoCo judge 尚未运行。详见[执行卷宗](experiments/phase1_fresh_sz_model4000_q50_20260713.md)。
- Native MuJoCo feasibility/implementation 已确认为 P0，但不阻塞几天内 `Gate3-D0`。off-main
  preflight `6e5fce3` 的 63 项 focused test、顶层 `468 passed, 9 skipped` 和七个 false 授权位
  证明 fail-closed；red team 同时抓出 action trace、source alias/exec、strict JSON、MJCF
  `strippath` 四个高优先级正确性缺口，所以当前 `NO-MERGE`。single-env core 未来还必须过
  N=1/8/32/64 与 48 小时留 30% 余量的吞吐继续门。它不是 trainer、`VecEnv`、PPO smoke 或训练结果，详见
  [实验卷宗](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md)。
- 正手拍面复核纠正了“所有 seed 都约 170°”的旧说法：model-2000 seed1/2/3 raw-A 误差为
  `171.10/172.94/173.39°`，seed4 没有正手 exact strike；解析回球器的 `orient_normal`
  可能抹掉正负号。signed-face 诚实门通过前，旧解析高分不用于晋级；详见
  [拍面符号卷宗](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。
- 连续拍等待/恢复设计经原文和现役代码复核后收紧：T0 按周期换题，T1 只改事件驱动结构并冻结
  reward，T2 才允许 learned shaping；随机到球先作为环境轴。若 T1 失败，先做平衡债/ready
  potential 的配对 `2^2`，第三 critic 只有独立校准并通过隔离 q50 后才能进入 `2^3`。
  这次只收紧文档设计边界；现有 machine prereg/validator/operation 仍固定旧三 reward/full `2^3`，
  必须另做内容寻址同步后才能点火。没有训练、simulator、Pod 或真机行为结果；详见
  [连续时序审计](research/phase1_continuous_rally_timing_2026-07-11.md)。
- 新动作的冻结站位 `0/64` 不再解释为“动作无效”：正式问题是动作自身安全触球流形 × 适配来球/动作题族
  × 合法整轨 `SE(2)` 站位。反手拉 B/C 仍只到重定位候选，挡球需另出题；没有重跑 screen。
- `main@3c7e507` 先补回了缺失的 INDEX 和实验账骨架；本分支把它升级为中文一站式路由、
  术语人话表、逐实验卷宗、精简 NOW/TIMELINE/PROGRESS、唯一队列和算力纪律。合入 `main`
  前，新版 NOW 仍只是一份提案。本次文档迁移没有运行训练、simulator、Pod 进程或真机。

## 2026-07-12

- 完成 native MuJoCo `Trainer-v0` 只读 preflight：现役 vendor main 的 sim loop 没有球/球台/网，
  所以首卷只做单拍 balance/strike-state fine-tune；reward 用独立 replay oracle；warm start 只载 actor，
  critic/optimizer 全新。没有启动 backend、sim、Pod 或真机，详见
  [preflight](research/mujoco_training_v0_preflight_2026-07-12.md)。
