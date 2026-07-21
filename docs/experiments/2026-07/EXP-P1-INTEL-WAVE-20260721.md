# EXP-P1-INTEL-WAVE-20260721 — Wave Q 情报波：速度泛化 / 强 q_des 平滑 / 全身模仿 / 全关节 qdes barrier（8 臂）

- 状态：`preregistered`
- 阶段/轴：Phase 1 / 四条 2026-07-21 外部情报各买一条单变量消融臂（x2 父本）
- 集成小目标：用最小预算（每臂 4001 updates）把四条口头情报变成可拒绝的单变量证据，
  决定哪条值得进正式配方
- 人类负责人：Franco（情报来源与拍板）
- 执行者：Claude
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（prereg + 队列/渲染器/单测已落盘）。发射后升级路径同 push 波：
  逐臂 full-scene probe → `E3` runtime mechanics；终档 K100 同卷 → 单 seed 机制诊断
  （谱系 inexact，永为 diagnostic）；正名须 exact-lineage 重跑。
- 创建日期/最后复核日期：2026-07-21 / 2026-07-22（spdmix v2 改造落盘，两臂 blocked→ready）

人话导言：Franco 2026-07-21 带回四条情报，本波每条买一份最小证据。8 条臂 = 两个
`model_6700` 父本 `W/V`（`W`＝拍心优先×自由非击球臂的稳定父本，`V`＝拍速优先×强准备的
高摔倒父本）各配 4 档情报臂 `{spdmix, hstrong, fullbody, qbar}`，基础配方与正在跑的
24 格矩阵 [`w_c_s0`/`v_c_s0`](EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md)（C+S0：
`action_rate=-0.10`、slew hinge `0`、三个稳定机制 weight 全 `0`、测量 probe 全开）
**逐字相同**——那两格就是本波对照，不再买新对照。每臂 `4096 environments`、`seed=3`、
从各自 parent `model_6700` 续训 `4001` updates（预算刻意小于矩阵的 10001：情报波只买
方向，不买终局）。

## 动机（逐条对应四条情报）

1. **速度泛化是最高价值的泛化轴（→ spdmix）。** 来源：外部交流 + PNP 跑酷论文的证据
   ——速度维度的域随机化对运动技能迁移的收益最大。**战略声明（Franco 拍板）**：拍面、
   引拍、脚步等其他泛化轴**降级**，不在本波买臂；扩充行为覆盖走"**多录动作**"路线
   （真数据优先），不靠合成轴扩展。
   - **v1（在线变速，已被架构守卫拒绝——保留为历史）**：原设计把
     `task.motion.speed_scale_range` 从 `[1.0,1.0]` 原位替换为 `[0.8,1.2]`（每次挥拍
     在线采样播放速度）。planner task-revision 相位 governor 独占参考时钟，在线变速
     等于造第二个时钟，启动即 fail-closed（`mdp/commands.py:499-503` ValueError）。
     该拒绝是架构守卫按设计工作，不是 bug。
   - **v2（固定变速烤入片段列表——当前设计，2026-07-22 改造落盘后 ready）**：正确
     路径是把 TOPP 重定时的变速片段**离线烤入**（`bake_topp_strike_speed.py`：帧数/fps/
     触球帧/相位全不变、触球行逐位相同），以多 clip `motion_file` 列表混入训练（正手
     0.8/1.0/1.2 三档、反手 0.8/1.0/1.1 三档——反手 1.2 物理不可行，envelope
     `max_feasible_ratio=1.1931`，最近可行烤档 1.1）。每次挥拍对 clip 均匀随机采样、
     各 clip 原生 1.0x 播放，governor 仍是唯一时钟，无守卫冲突。曾被两道 2-clip
     硬绑定挡死（swing_sign 写死 clip0=正手其余=反手；schema-v3 题库按 2-clip 内容
     寻址 + motion SHA 对账 fail-closed，见
     [spdmix v2 可行性核查](../../research/spdmix_v2_multiclip_feasibility_20260722.md)）；
     **2026-07-22 最小改造已落盘**：`task.motion.clip_family_per_clip` 族表（缺席=
     现役所有臂行为逐字节不变，≥3 clip 缺表 fail-loud）＋题库族寻址与族级允许 motion
     SHA 列表（`rebind_question_bank_motion_family.py` 只扩 meta、题目张量逐字节不变），
     config/runner/tests 同步回填。剩余闸门只有一个：主控把改造合并进 `source.commit`
     并重钉 40-hex（checklist 第 3 条 grep 核对），之后 spdmix 两臂按原顺位首发。
2. **现役 q_des penalty 可能太小（→ hstrong 与 qbar 两档）。**
   - hstrong：把恢复窗 `processed_qdes_slew_hinge` 从矩阵 H 档的 `-0.25` 提到 `-1.0`
     （margin `0.85`、窗 `0.2–1.55 s` 与 H 档逐字同），同时 `action_rate=0`——
     检验"罚得不够重"这半句；矩阵在跑的 `w_h_s0`/`v_h_s0`（`-0.25`）天然是剂量对照。
   - qbar：Jiayi v14 用了 top-k qdes 限位 barrier，Franco 判断 **top-k 不必要**——
     全关节直接罚更简单也更稳。本波按去 top-k 语义新实现：全关节的 `q_des`
     距软限位不足 margin `0.08 rad` 即罚，逐关节有界 tail **求和不平均**（Franco：
     mean 是稀释器，单关节违规被 ÷31；sum 让违规几个关节就罚几份，这也是去 top-k
     的正当性所在；满违规单关节 tail≈1 → 每步约 `-0.65×dt`），weight `-0.65`。
     同时按 Franco"**把别的去掉**"裁定：qbar 臂把重叠最大的 raw `action_rate`
     从 `-0.10` 归零，barrier 是该臂唯一 qdes 惩罚（`joint_vel`/`joint_torques`
     量级 ~1e-4/1e-5，保持不动）。
3. **全身学习（全身模仿）可能很有用（→ fullbody）。** 这是 Franco **两阶段下肢方案
   的第一阶段：静止击球下肢全程软模仿＝真·全身全程模仿**——先学会"不动的"下肢全程
   跟参考。具体：把矩阵 S3 已实现的下肢姿态软模仿从试探剂量 `+0.5` 提到 `+2.0`
   （std `0.35`），并把支持窗从击球前 `0.3 s`/后 `0.4 s` 放开到全程（pre/post 各
   `10.0 s`，gate 语义是 pre 窗 `[0,pre_s]` ∪ 同拍击后 `[0,post_s]`，10 s 覆盖整个
   episode 相位）。第二阶段是横移拼接：引拍时跨步（prepare）、击球窗双脚钉死
   （strike）、随挥收腿回位（recover），语义合同见
   [`docs/interfaces/stroke_footwork_composition.md`](../../interfaces/stroke_footwork_composition.md)。
   S3 格（`w_*_s3`/`v_*_s3`）作剂量+窗宽对照（S3 是窄窗 `+0.5`，本臂是全程 `+2.0`，
   判读按"方案对方案"读，不拆键）。
4. **caveat（只记录，不做臂）：Jiayi 的训练分布还没调好。** Jiayi 自述其分布尚未调
   优，故其 v14 一切配方证据（含 qbar 的原始出处）只作方向参考，不能当剂量依据；
   qbar 臂判读时必须带上这条 caveat。本波不复刻 Jiayi 的其它配方成分。

## 对照声明（不重复买）

矩阵在跑的 `p1btm_w_c_s0_seed3_20260720` / `p1btm_v_c_s0_seed3_20260720` 与本波基础
配方逐字相同（C+S0、`speed_scale_range=[1.0,1.0]`、slew hinge `0`、下肢模仿 `0`、无
barrier），直接当对照；hstrong 另有矩阵 H 档、fullbody 另有矩阵 S3 档当剂量对照——
**三个对照族全部复用矩阵既有格，本波零对照臂**。对照跨 queue：两边 source commit
不同时须 diff 训练路径无行为性改动并记录结论，否则对照失效、必须自买同 commit 对照。

## 可证伪假设（每臂一条，均在各自 parent 内与矩阵对照同 absolute milestone 比较）

1. **spdmix（速度泛化，v2 烤入列表；改造已落盘，ready）**：烤入变速
   片段混合（正手 0.8/1.0/1.2、反手 0.8/1.0/1.1 六 clip 均匀采样）训练后，终档 K100
   同卷回球不低于对照 `−3` 题、摔倒不增，且（若变速考卷可用）变速条件下优于对照。
   若回球掉超过 `3` 题或摔倒显著增加→拒绝"速度混合免费"，速度泛化需另立课程化方案。
   前置条件（2026-07-22 已满足本地部分）：clip_family + 题库族重绑改造已落盘并单测
   通过、资产已上传两 pod 并 SHA 对账；仅剩主控合并进 `source.commit` 并重钉 40-hex。
2. **hstrong（强 q_des 平滑）**：`-1.0` hinge（无 action_rate）相对对照 processed-q_des
   tail 显著收紧、摔倒不增、回球不低于 `−3` 题；且相对矩阵 H 档（`-0.25`）呈剂量响应。
   若回球明显塌掉→拒绝"现役 penalty 太小"，是"罚法不对"而非"罚得不够"。
3. **fullbody（全身全程模仿＝两阶段下肢方案第一阶段）**：`+2.0` 全程窗下肢模仿相对
   对照摔倒下降且回球不低于 `−3` 题；与 S3（窄窗 `+0.5`）比按"方案对方案"读方向。
   若 `+2.0` 全程反而僵化（回球塌）→拒绝"越重越全程越好"，回到 S3 窄窗剂量带，
   两阶段方案的第一阶段改小剂量重试。
4. **qbar（全关节 qdes barrier，sum 聚合）**：`-0.65/0.08`（去 action_rate）使
   `q_des CLAMP ACTIVE` 行数与限位撞击显著下降、摔倒不增、回球不低于 `−3` 题。
   否则拒绝；判读必须带 Jiayi 分布未调好的 caveat（见动机 4），不得据本臂反推
   Jiayi v14 整体好坏。

physical-fall 硬失败不能由更高 reward、composite 或更平滑的 action 抵消。

## 冻结的 setting

| 字段 | 冻结值 |
| --- | --- |
| 队列/渲染器 | [`configs/phase1_intel_wave_20260721.yaml`](../../../configs/phase1_intel_wave_20260721.yaml)、[`scripts/run_phase1_intel_wave_queue.py`](../../../scripts/run_phase1_intel_wave_queue.py)（lean runner 纪律与矩阵/push 波同款：`--plan` 计划表＋`--checklist` 核对单＋`--render-stage probe\|science --render-job <id> --pod <pod> --gpu <n>` 逐字渲染人工核对；不写死 pod/gpu，渲染时注入；renderer 绝不 SSH/发信号/写远端） |
| 基础配方 | 与矩阵 `C+S0` 格逐字相同：base_overrides、planner revision、各 parent recipe_overrides、`action_rate_weight=-0.1`、slew hinge `0`（非 hstrong 臂）、三个稳定机制 weight `0`（非 fullbody 臂）＋全参数显式写、probe 全开（单测与矩阵 yaml 交叉断言防漂移） |
| spdmix 档 | **v2＝烤入变速六 clip 列表，2026-07-22 改造落盘后 ready**：`motion_file=[fh0.8,fh1.0,fh1.2]` + `motion_file_2=[bh0.8,bh1.0,bh1.1]`（TOPP 烤入、帧数/fps/触球帧/相位不变，逐 clip 资产表见下节）、base 原位扩位 `strike_phase_per_clip=[0.471,0.471,0.471,0.338,0.338,0.338]`、`mount_normal_sign_per_clip=[1.0,1.0,1.0,-1.0,-1.0,-1.0]`、extra 带 `++task.motion.clip_family_per_clip=[forehand,forehand,forehand,backhand,backhand,backhand]`、题库换按族重绑版 `s1_..._family6_rebound.npz`、`speed_scale_range` **保持** `[1.0,1.0]`（governor 单时钟不受扰；渲染器断言 8 臂全 `[1.0,1.0]`，`[0.8,1.2]` 任何臂出现即拒绝）。历史：v1 在线 `[0.8,1.2]` 替换启动即被 `commands.py` 守卫拒绝；两道 2-clip 硬绑定（[可行性核查](../../research/spdmix_v2_multiclip_feasibility_20260722.md)）已由 clip_family 查表＋题库族寻址改造解除（缺席配置行为逐字节不变） |
| hstrong 档 | `action_rate_weight=0.0` + `processed_qdes_slew_hinge_weight=-1.0`，margin `0.85`、恢复窗 `0.2–1.55 s` 三行逐字 = 矩阵 H 档 |
| fullbody 档 | stability 串 = 矩阵 S0 三行替换：`lower_body_pose_imitation_weight` `0.0→2.0`（std `0.35` 不动）＋支持窗 `support_pre_s/post_s` `0.3/0.4→10.0/10.0`（≈ 全程；两阶段下肢方案第一阶段），单测断言恰好只差这三行 |
| qbar 档 | `++task.rewards.qdes_limit_barrier_weight=-0.65`、`++task.rewards.qdes_limit_barrier_margin=0.08`（全关节、无 top-k、逐关节有界 tail **求和**不平均）＋时序串把 `action_rate_weight` `-0.1→0.0`（Franco："把别的去掉"，barrier 是该臂唯一 qdes 惩罚）。键名按任务书冻结；本地工作树冻结时 grep 无该 wiring，故 `qbar_contract.qbar_wiring_confirmed=false` **锁死 qbar 两臂渲染**（其余六臂不受影响），主控合并 wiring、逐字核对远端 train.py 白名单后才翻 true |
| 无关键面剔除 | 本波不含任何 `task.push.*` / `task.force_push.*` 键（缺席＝逐字节 no-op）；渲染器对任何臂出现这两族键直接拒绝 |
| 对照 | 矩阵 `p1btm_w_c_s0_seed3_20260720` / `p1btm_v_c_s0_seed3_20260720`（不重复买）；剂量对照：H 档（hstrong）、S3 档（fullbody） |
| parent SHA 钉死 | W checkpoint `2caab3dd...fcce`、V checkpoint `ad901910...2716`（与矩阵同两份 `model_6700`）；发射前远端 `sha256sum` 逐一比对 |
| source | checkout 复用 push 波的 `/workspace/codexschema/nohope_push_20260721`；clean detached exact commit；config 出厂占位 `PENDING_EXACT_COMMIT`，占位符下渲染被拒绝（单测覆盖），qbar wiring 合入 main 后才填 40-hex |
| namespace | `/workspace/codexschema/phase1_intel_wave_20260721`（fresh、no_clobber、no-retry） |
| parent/seed | `W@model_6700` 与 `V@model_6700`，全部 8 臂 `seed=3`，完整恢复 policy/value/optimizer/normalizer |
| 预算 | 每臂续训 `4001` updates（absolute 至 `model_10700`）、save/100；观察点 offsets `+200/+500/+1000/+2000/+4000`（absolute `6900/7200/7700/8700/10700`）——量尺口径同矩阵，预算刻意缩短 |
| watchdog | boot 停滞 `1800 s`、首迭代后停滞 `900 s` 才判死；不用 `launch_kit_training_locked.sh`（180 s stale 门是 v8/v9 死因）；重试只许逐字 `_r2` 一次 |
| 谱系 | `checkpoint_allow_contract_mismatch=true`；W/V contract 有意 mismatch，所有后代只作诊断、永久 formal-exact-ineligible |

## spdmix v2 资产表（2026-07-22 上传两 pod，SHA 逐一对账）

远端目录（两 pod 同路径）：`/workspace/codexschema/phase1_intel_wave_20260721/assets/topp_speed/`；
逐文件 SHA-256 与上传对账见同目录 `upload_manifest_spdmix_v2_20260722.json`
（pod1 162.43.172.171 与 pod2 162.43.172.181 `sha256sum` 全部 14 个文件与本地逐一相等）。
每份烤入 npz 旁带同名 bake manifest（`.json`，含 `contact.row_bitwise=true`、
`feasibility.verdict=feasible`、TOPP 判卷逐关节利用率）。

| clip（motion_file 顺序） | 家族 family | 速度档 | strike_phase | mount_sign | 帧数/触球帧 | npz SHA-256 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `hope_forehand_v4rg_speed0p80.npz` | forehand | 0.8 | 0.471 | +1.0 | 141 / 66 | `d1c014da7e25c8a72bf1acd16d131d69547ded3234d4a8cb5a6837d22bf677de` |
| `hope_forehand_v4rg_speed1p00.npz` | forehand | 1.0 | 0.471 | +1.0 | 141 / 66 | `2c2521fdd5bb51c78cbc775d073b3b17c8d591bd49be5caae956757baa4f27e6` |
| `hope_forehand_v4rg_speed1p20.npz` | forehand | 1.2 | 0.471 | +1.0 | 141 / 66 | `52fc43a50b2bdf07469237beae81738411b9172699c92bc3c8a1fa25be050420` |
| `hope_backhand_v4rg_speed0p80.npz` | backhand | 0.8 | 0.338 | −1.0 | 134 / 45 | `7d244909dc33a1c60ca4e53cf13b37c8b3bfc7c2c99f286e136dfcbd6ae59fd2` |
| `hope_backhand_v4rg_speed1p00.npz` | backhand | 1.0 | 0.338 | −1.0 | 134 / 45 | `edee029a2b629481367d5e746a595cc87fc6195c43e092c5070bc632e87cc803` |
| `hope_backhand_v4rg_speed1p10.npz` | backhand | 1.1 | 0.338 | −1.0 | 134 / 45 | `11c459e2d9a9101f94ec3e61d2463e877d7cccc552c2ebfad081ef1af547235a` |

题库（仅 spdmix 两臂）：`s1_v4rg_runtime_order_schema3_train_882fea4_family6_rebound.npz`，
SHA-256 `835ddd89e02eb98ef30fd60317ca97f6288e502fb41a7e15ec6fba690ffafb60`——由
`rebind_question_bank_motion_family.py` 从现役 train 题库
`s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz`
（`3a9d8851c1c0b13ef82f58228ea1cf83213157c70d72daa514f1bed3a3885b71`）生成：题目张量
逐字节不变，只扩 meta 的族级允许 motion SHA 列表（每族＝cal 原件＋三个烤入，共 4 个；
正手 cal `f2cb2d9f…1687`、反手 cal `17225533…7534`）。重绑对账
`s1_v4rg_runtime_order_schema3_train_882fea4_family6_rebind.json`（单层内容 SHA，
含逐烤入 `row_bitwise`/frames/contact_frame/speed_ratio 登记与 loader 整卷回验）。
其余六臂继续用 cal 原件＋现役 train 题库，路径与 argv 逐字节不变。

## 8 臂表与唯一 run names（表序 = 冻结 launch_order）

run name 模式为 `p1iq_{w|v}_{spdmix|hstrong|fullbody|qbar}_seed3_20260721`。

| # | 人话 — `run_name` | 情报臂改动（相对矩阵 C+S0 对照的唯一差异） |
| ---: | --- | --- |
| 1 | W 速度泛化混合 — `p1iq_w_spdmix_seed3_20260721` | v2＝烤入变速六 clip 列表（正手 0.8/1.0/1.2＋反手 0.8/1.0/1.1，`speed_scale_range` 保持 `[1.0,1.0]`）＋clip_family 族表＋按族重绑题库；**ready**（2026-07-22 改造落盘、资产上传对账完；发射前主控须把改造合并进 `source.commit` 重钉 40-hex。v1 在线 `[0.8,1.2]` 被 governor 守卫拒绝＝历史） |
| 2 | V 速度泛化混合 — `p1iq_v_spdmix_seed3_20260721` | 同上 |
| 3 | W 全身全程模仿 — `p1iq_w_fullbody_seed3_20260721` | `lower_body_pose_imitation_weight 0→2.0`＋支持窗 `0.3/0.4→10.0/10.0` 全程（两阶段下肢方案第一阶段） |
| 4 | V 全身全程模仿 — `p1iq_v_fullbody_seed3_20260721` | 同上 |
| 5 | W 强 q_des 平滑 — `p1iq_w_hstrong_seed3_20260721` | `action_rate -0.1→0`＋`slew hinge 0→-1.0`（时序正则替换；对 H 档单变量） |
| 6 | V 强 q_des 平滑 — `p1iq_v_hstrong_seed3_20260721` | 同上 |
| 7 | W 全关节 qdes barrier — `p1iq_w_qbar_seed3_20260721` | `qdes_limit_barrier weight -0.65 / margin 0.08`（sum 聚合）＋`action_rate -0.1→0`（barrier 是唯一 qdes 惩罚；渲染待 wiring 闸门） |
| 8 | V 全关节 qdes barrier — `p1iq_v_qbar_seed3_20260721` | 同上 |
| — | 对照（矩阵在跑，不发射） — `p1btm_w_c_s0_seed3_20260720` / `p1btm_v_c_s0_seed3_20260720` | 无 |

`launch_order` 冻结为 `w_spdmix → v_spdmix → w_fullbody → v_fullbody → w_hstrong →
v_hstrong → w_qbar → v_qbar`（价值排序：最高价值轴先买；qbar 排最后，等 wiring 闸门）。
**spdmix 两臂 2026-07-22 起解锁回首位**（v2 改造落盘＋资产上传对账完，见冻结 setting 表
spdmix 行与资产表）；唯一剩余前置是主控把改造合并进 `source.commit` 并重钉 40-hex
（checklist 第 3 条），未重钉前空槽轮到时顺延从 `w_fullbody` 起拉，重钉后按原顺位补发。
机器真源是 queue config，其 8 条 job 的 override 与本表不符时必须先修订本记录再渲染命令。
`W` 与 `V` 配方不同，一切比较都在各自 parent 内进行。

## 发射策略：空槽即填，不建专属池

与 push 波共用同一单一队列纪律：不给 GPU/pod 分角色，矩阵/push 波谁毕业谁腾卡，空槽
出现时按跨波总队列的人工裁定拉最前面就绪的臂（本波内部顺序即上表）；每卡并发上限沿用
"同卡 compute PID < 4"预检（命令内已带）。每臂先跑 full-scene probe（`4096 env x 24
step x 2 update`，独立 `probes/` namespace），自然退出、`run.log` 出现 `Learning
iteration`、无 fatal、`model_6701.pt` 存在后才允许发 science。同 pod 内 boot 串行
（kit_boot_lock 持锁），相邻 launch 错峰 `>= 60 s`。日志摘要抓异常不抓预期：WARN 行
与 `q_des CLAMP ACTIVE` 行必须全部进摘要（qbar 臂的 CLAMP 行数本身就是判读量之一）。

## 判据与量尺（同矩阵）

1. **Isaac 内先行指标**：fall、ready tilt、processed-q_des tail、completion/legal-return
   ——固定 100-update 窗对 exact ledger counter 求和再除，不得插值、改窗或事后挑点；
   口径逐字沿用[矩阵量尺](EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md#预算与观察点)。
   spdmix 臂披露：其训练窗指标是在变速环境里测的，与恒速对照的跨臂比较只作方向参考。
2. **主判据＝终点同卷**：`model_10700` 终档 K100 同卷（100 题、正反手各 50）对每臂与其
   matched 矩阵对照的**同 absolute milestone**（`10700`）档判卷；同卷同 scorer 不混卷。
   假设 1–4 的门全部在这张卷上裁决。
3. **观察点分工**：`6900/7200/7700` 只判崩溃/合同/接线（qbar 臂加验 barrier 探针与
   CLAMP 行数量级；spdmix 臂改验 6-clip 装载回显与 clip 采样熵/份额指标——v2 无在线
   speed_scale 采样）；`8700` 看中段；`10700`
   出单 seed 结论。不因稀疏 reward 早期为零而停。
4. **谱系限制**：全部 8 臂永久 formal-exact-ineligible，结果只是 diagnostic；胜者臂要
   正名必须到 exact-lineage 链单变量重跑。单 seed 3 只做 screening，胜者另立预注册补
   seed。最终行为裁判仍是 vendor MuJoCo 同卷。

## 停止规则（与矩阵相同）

只有三类情况允许停臂（记录原因与最后 checkpoint，其余一律跑满预算）：

1. **NaN/OOM 不可恢复**；
2. **lineage 错误**（parent 错、情报键漂移、`training_contract_lineage_exact` 意外为 `1`）；
3. **被同 parent 支配**：同 parent 内存在另一臂（含矩阵 `C+S0` 对照），fall 与击球
   （completion 与 legal return）全部严格更优且持续 `>= 4000` iterations（本波预算即
   4001，实际上意味着几乎不裁、跑完为主）。

不允许因"早期难看"、稀疏 reward 为零或跨 parent 比较而停臂。停止执行只能按
[RunPod 纪律](../../operations/run_on_runpod.md)对 exact PGID 处置；禁止 `pkill -f`。

## 风险与已知边界

- **qbar wiring 未落盘（冻结时 grep 已证）**：本地工作树 train.py 无 `qdes_limit_barrier`
  ——键名按任务书冻结写入 config，`qbar_contract.qbar_wiring_confirmed=false` 锁死 qbar
  两臂渲染；单测断言"wiring 未落盘时闸门必须关着"。主控合并 wiring、核对键名逐字一致、
  冻结新 40-hex commit 后翻 true；exact 语义以实现+单测为机器真源，与本文不符时先修订
  本记录再渲染。
- **hstrong 是"时序正则替换"不是加法**：相对 C 对照同时动了 `action_rate`（-0.1→0）与
  hinge（0→-1.0）两个键——这与矩阵 N/C/H 轴的定义一致（每档一个完整时序方案），对
  H 档（同为 action_rate=0）则是纯 hinge 剂量单变量。判读时按"方案对方案"读，不拆键。
- **qbar 同理是"qdes 惩罚方案替换"**：相对 C 对照同时动了 barrier（新增，sum 聚合
  `-0.65/0.08`）与 `action_rate`（-0.1→0，Franco："把别的去掉"）——barrier 是该臂唯一
  qdes 惩罚，判读按"方案对方案"读，不拆键；与 hstrong（同为 action_rate=0）可读出
  "barrier vs 强 hinge"的方案对比。
- **fullbody 是"剂量＋窗宽"双动**：相对 S3 同时动了 weight（+0.5→+2.0）与支持窗
  （0.3/0.4→10.0/10.0 全程）——这是两阶段下肢方案第一阶段的定义（全程软模仿），
  S3 只作方向参考，不构成纯剂量单变量。
- **Jiayi 分布 caveat**（动机 4）：qbar 结果无论正负都不外推到 Jiayi v14 整体。
- **对照跨 queue**：对照/H 档/S3 档全在矩阵 queue（20260720 commit），本波 commit 不同
  ——发射前 diff 训练路径无行为性改动并记录，否则对照失效。
- **spdmix v2 与 motion 时长交互**：烤入片段帧数不变，单次挥拍时长/触球帧与恒速对照
  逐帧一致（v1 在线变速才会改时长——已否决）；probe 收口改核对 6-clip 装载回显、
  strike/mount 六位表与 clip 采样份额无异常。
- **spdmix v2 硬绑定已解除（2026-07-22），剩 commit 重钉**：两道 2-clip 硬绑定
  （[可行性核查](../../research/spdmix_v2_multiclip_feasibility_20260722.md)）由
  `clip_family_per_clip` 查表＋题库族寻址/族级允许 SHA 列表改造解除，铁律"缺席配置
  行为逐字节不变"有单测背书；资产（六烤入＋族重绑题库）已上传两 pod 并对账（见资产表）。
  当前 `source.commit`（`4624c824`）**不含**该改造：主控合并并重钉 40-hex、按 checklist
  第 3 条 grep 远端 `train.py` 白名单确认 `clip_family_per_clip` 后才发 spdmix 两臂；
  在旧 commit 上强发会在 boot 期 fail-loud（未知 motion 键/6-clip 缺族表当场报错）。
- **预算只有 4001**：慢显影机制（尤其 fullbody 大权重的前期扰动）可能在 `10700` 还没
  收敛完；结论按"方向 + 是否值得进正式配方"读，不按终局读。
- **诊断谱系**：全部后代 formal-exact-ineligible；任何"采用"都要走 exact-lineage 重跑。

## 运行表

| 运行 | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| 8 臂（上表 #1–#8） | `preregistered / not launched` | W/V `model_6700` / seed3 | E1 prereg | 无 | qbar wiring 合入 main、占位 commit 解锁、`--checklist` 全过、`origin/main` NOW 认领、空槽出现且逐臂 probe 过后才发对应臂；qbar 两臂另需 `qbar_wiring_confirmed=true`；**spdmix 两臂 ready（v2 改造 2026-07-22 落盘、config/runner/tests 已回填、资产已上传对账），另需主控把改造合并进 `source.commit` 并重钉 40-hex**（checklist 第 3 条 grep 核对） |

## 分动作成绩表

| 动作 | 一次挥拍物理不摔 | 一次挥拍击球 | 一次挥拍上台 | 连续挥拍物理不摔 | 连续挥拍击球 | 连续挥拍上台 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 反手 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

"物理不摔"将使用 `physical_fall_count`（all-attempt 分母）。本波是 single-swing
continuation 诊断，不声称连续恢复；即使 Isaac/K100 改善，仍须独立 vendor MuJoCo/Gate3
行为卷。

## 决定

- 决定：`pending`（preregistered，未发射；spdmix 两臂 2026-07-22 起 ready，待主控
  合并 v2 改造并重钉 `source.commit`）
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：主控把 spdmix v2 改造（clip_family＋题库族寻址＋rebind 脚本）
  合并进 main、重钉 `source.commit` 40-hex 并 diff 对照 commit 无行为性改动；
  `origin/main:docs/NOW.md` 完成认领；`--checklist` 全过（含第 3 条 spdmix grep/资产
  SHA 对账）、空槽出现、逐臂 probe 核对通过后按冻结 `launch_order` 逐臂发射
  （spdmix 两臂居首位）；`10700` 终档 K100 同卷裁决假设 1–4；胜者臂到 exact-lineage
  链正名。

## 复现与证据

队列与渲染器见冻结 setting 表首行；渲染出的 SSH 命令一律人工核对后执行。本记录当前
没有任何 SSH、claim、checkpoint 或行为结果；judge、第二 seed、部署与真机均未授权。
