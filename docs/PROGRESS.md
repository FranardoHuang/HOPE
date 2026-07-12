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
  `prepared_not_started/jobs_started=0/auto_start=false`，没有 run、judge、新分、trainer signal
  或真机动作；还缺可审计的常驻父 supervisor。详见
  [Fresh SZ 稳定性实验](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。
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
