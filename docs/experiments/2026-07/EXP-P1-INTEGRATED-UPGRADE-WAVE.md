# EXP-P1-INTEGRATED-UPGRADE-WAVE — 集成升级波（3 臂，2026-07-23）

- 状态：`preregistered`（action_acc 闸门锁全波 + franco 闸门锁 combo_franco，未渲染任何命令）
- 阶段/轴：Phase 1 / 集成合体波——把各单变量波里已证明有用/Franco 拍板值得合体的升级
  一次性合进一个配方 go for a try（**不做单项归因**，归因永远看各单变量波）
- 集成小目标：回答一个问题——"全部升级合在一起，站立击球配方是变强还是互相打架？"
  三个谱系各买一臂：续训（可直接对 `w_c_s0`）、fresh 粗糙地（绝对水平）、Franco 动作
  （换反手素材的可训性）
- 人类负责人：Franco（拍板与 07-23 审查五变更+两追加）
- 执行者：Claude
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（prereg + 队列/渲染器/单测已落盘：三件套 55 项测试全绿 +
  CGF/intel/push 三波回归 186 项全绿）。发射后升级路径同 CGF 波：逐臂 probe → `E3`
  runtime mechanics；终档 K100 同卷 → 单 seed 机制诊断（谱系 inexact，永为
  diagnostic）；正名须 exact-lineage 重跑。
- 创建日期/最后复核日期：2026-07-23 / 2026-07-23

人话导言：CGF 波（抖动-地面-脚部消融）、push 波（推撞鲁棒性）、intel 波（qbar barrier）
各自买了单变量臂，Franco 拍板不等全部读数收口，把其中"已证明有用/证据链最硬"的升级
一次性合体试一把。合体清单与各自证据：

| 升级项 | 剂量 | 证据（引用） |
| --- | --- | --- |
| action_rate -0.1→-0.2 | 全臂 | jiayi V14 报告的关节抽动＝现役 -0.1 太小的实证；[站立异响 E1 复核 PDF](../../research/agibot_a3_standing_chatter_evidence_ranked_review_2026-07-22.pdf) 确认 action-rate 偏弱成立；CGF `ar02` 臂同档单变量在测。反向情报（jiayi V15 降到 -0.01 靠投影约束、mjlab 与我们同值 -0.1）照记 |
| 全程高摩擦 静[1.0,1.6]/动[0.8,1.2] | 全臂 | CGF `grip` 臂假设（别的队加了摩擦+不平整后脚不搓地）；Franco 07-23 变更①：0.6 下界还是太低——摩擦当已知量拉高，失稳压力交给 push/凹凸轴 |
| mjlab 档①三项全收 | 落地罚全臂；抬脚罚只 fresh；action_acc 全臂 | [mjlab 采纳审计](../../research/mjlab_reward_adoption_20260722.md) 档①【现在就要】：落地罚 -3e-3 @300 N（无量纲超阈倍数，mjlab -1e-5/N 等效剂量）；抬脚罚 -0.01 @0.15 m（凹凸地面就是为了逼抬腿——落地罚+抬脚罚成对，target 0.15 = hope_rewards"真要抬腿跨步"档）；action_acc -0.05（二阶平滑=抖动的正交新轴，剂量=action_rate 的 1/4，落在该文档"1/5~1/2 先小"带内；**源码未接线，闸门锁死**） |
| qbar 全关节 qdes 限位 barrier -0.65 / margin_frac 0.08 | 全臂 | 别人经验≈限位前 8-10% 就罚（有用）；我们 V14 验证档（margin_frac 是行程比例，07-22 语义裁定）；intel 波 `v_qbar` 臂判卷反手 1.00 佐证不压击球。与 action_rate=-0.2 并存——intel 波是单变量设计才互斥 |
| 速度推 ±0.35 m/s 每 5-15 s | 全臂 | push 波 `w_p035` 臂键面逐字（PACE 与 BeyondMimic 之间档位；±0.5 在 W 父本上零摔的读数说明 0.35 是安全档） |
| 同冲量力推 68 N x 0.30 s | 全臂 | push 波 `w_f035` 臂键面逐字（pelvis link 原点、Δv_equiv=0.35005 m/s @ 58.27723163 kg，与速度推同冲量对表）；Franco 07-23 变更②：两组事件并存 |
| 凹凸地形 2-6 cm | 只 combo_fresh | CGF `rough` 臂同款（平地过拟合假设；fresh-from-random 铁律） |
| 反手动作换 franco_bh_loop_b | 只 combo_franco | [动作资产看板](../../research/franco_motion_asset_board_20260722.md)：151f @50Hz、唯一三证（L0/vendor L1/整轨桌网）+六轴 phase-scan 100% @0.59-0.61；欠 grip 标定→锚入册→按族重绑（franco 闸门三前置） |

父本只用 `W`（拍心优先×自由非击球臂，`model_6700`）；对照＝矩阵
[`w_c_s0`](EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md) 不重复买。续训臂预算 `13301`
updates（6700+13301≈20001，对齐 Jiayi 2万-2.5万 目标下沿）；combo_fresh 独立 `20001`。

三件套：[configs/phase1_integrated_upgrade_wave_20260723.yaml](../../../configs/phase1_integrated_upgrade_wave_20260723.yaml)
+ [scripts/run_phase1_integrated_upgrade_wave_queue.py](../../../scripts/run_phase1_integrated_upgrade_wave_queue.py)
+ [tests/test_run_phase1_integrated_upgrade_wave_queue.py](../../../tests/test_run_phase1_integrated_upgrade_wave_queue.py)。

## 3 臂假设与判读标准

| 臂 | 改动（人话） | 假设 | 判读标准 |
| --- | --- | --- | --- |
| combo_franco | combo_resume + 反手动作（motion_file_2）换 franco_bh_loop_b（Franco：换动作是全面升级的优先消融，launch_order 排最前） | Franco 反手拉 B 在扭矩/存活全面≥ v4rg，配上升级包应能学出击球 | 对 combo_resume 同里程碑：反手 hit/pass/composite（franco 反手族重绑题库判卷）；正手侧不应受影响（正手素材未动）。闸门未开前不发射 |
| combo_resume | C+S0 + 上表全部升级（无凹凸/抬脚罚），W model_6700 续训 13301 | 升级互不打架时，抖动/失稳代理应降、击球组不掉超噪声带 | 对 `w_c_s0` 同里程碑：击球组（pass/composite/virtual return）差 ≤ 噪声带且失稳代理（fall 率、settle 探针、slew 探针尾部）下降 → 集成配方候选；击球塌了 → 回单变量波读数定位元凶，本波不承担归因 |
| combo_fresh | 同 combo_resume + 凹凸 2-6 cm + 抬脚罚，fresh-from-random 20001 | 粗糙地+全套护栏能从零学出站立击球 | 只看绝对水平：粗糙地上 pass>0 + fall 率；与 `w_c_s0` 对比仅方向参考（谱系不同必须带 caveat） |

judge/K100 同卷判读用现役 signed-face 判分器（combo_franco 反手侧须先有按族重绑的
franco 题库）；单臂结论只升到"方向证据"，胜者配方须另在 exact-lineage 重跑才能进正式
配方。**combo 臂赢了也不能归因单项**——单项归因永远看 CGF/push/intel 各单变量波。

## 比值审计表（Franco 比值守卫；机器可校验部分在 YAML `reward_budget_contract`）

击球组每步收入（exp 核∈[0,1]，挥拍窗内）＝ **17/7/5/5/10，一动不动**：
racket_position 17 + racket_velocity 7 + racket_normal 5（连续追踪，显式钉在 W 配方串）
+ racket_strike_success 5（稀疏触球步）+ racket_progress 10（渲染器禁止这两键出现任何
override＝源码默认）。

| 新增负项 | 剂量 | 类型 | 每步量级（静态论证） |
| --- | --- | --- | --- |
| action_rate | -0.2 | 连续 | 现役 -0.1 已在 24 格矩阵全谱系全程跑过、击球未被压制（w_c_s0 台账）；本剂量=现役 2 倍（jiayi V14 关节抽动=偏弱证据） |
| action_acc | -0.05 | 连续 | 二阶差分量纲大于一阶，故取 action_rate 的 1/4（mjlab 采纳文档"1/5~1/2 先小"带内）；源码未接线，实测待 probe |
| foot_soft_landing | -0.003 @300 N | 稀疏（first-contact 步才计费） | 单脚封顶 3、双脚合计 ≤6 → 每步最坏 \|-0.018\|；站立击球 first-contact 稀少 |
| foot_clearance（只 fresh） | -0.01 @0.15 m | 稀疏（腾空脚才计费） | \|脚高-0.15\|×水平脚速×2 脚，最坏量级 ≈0.006/步；站立支撑常态 0 |
| qdes_limit_barrier | -0.65 / margin_frac 0.08 | 稀疏（贴限位才付费） | q_des 离限位 >8% 行程时恒 0；"贴限位才付费"，常态 0（v_qbar 判卷反手 1.00 佐证不压击球） |

**预算规则**：五项新增负项每步合计量级 ≲ 击球组每步收入 × 1/3。静态论证之外必须实测：
probe + 首个里程碑（续训 `model_6900` / fresh `model_200`）读 tensorboard 逐项 episode
均值核对 `(|action_rate|+|action_acc|+|foot_soft_landing|+|foot_clearance|+|barrier|) /
(position+velocity+normal+strike_success+progress) ≤ 1/3`，超限该臂停发 science/停训
裁决并记录读数（稀疏项 episode 均值异常大先查激活探针再谈停训）。

**软惩罚组不再叠加**（penlight 教训：软限制压制模仿→学不会击球）：face -0.4 /
foot_ori -0.3 / upright -1.0 全额不动，蹭滑/拖脚/挥拍前脚滑键一个不带（渲染器断言）。
**速度推/力推是事件不是 reward**，不进本表。

## Run table（发射后逐臂回填）

| job | run_name | stage | pod/gpu | 发射时刻 | 终档 | 里程碑成绩 | 比值实测 | 处置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combo_franco | p1iu_combo_franco_seed3_20260723 | — | — | — | — | — | — | 未发射（franco 闸门三前置未交付） |
| combo_resume | p1iu_combo_resume_seed3_20260723 | — | — | — | — | — | — | 未发射（action_acc 闸门锁全波） |
| combo_fresh | p1iu_combo_fresh_seed3_20260723（fresh） | — | — | — | — | — | — | 未发射（同上） |

## 依赖（全部满足前渲染器 fail-closed，不出任何 SSH 命令）

1. **action_acc_contract.wiring_confirmed 翻真（锁全部三臂）**：mjlab 档①第三项
   `task.rewards.action_acc_weight` 源码未接线（2026-07-23 grep 钉住 commit 的
   train.py 白名单无此键）。谁实现谁补单测（照 foot_soft_landing 先例：默认 weight=0
   字节等价、量纲断言、复位后前两步不计费），合入 main 后**重钉 source.commit 为新
   40-hex**、远端白名单核对一致再翻 true。
2. **franco_contract.wiring_confirmed 翻真（只锁 combo_franco）**：
   franco_pipeline_20260722 战役交付三前置——grip 标定 bake（B 的 phase-scan 现为未
   标定拍面口径）+ 空挥视觉锚点入册（PROBE 件不得注册训练）+ 按族重绑题库（franco
   反手族替换 v4rg 反手绑定）。三者齐了把 `assets.motion_backhand_franco` 从占位符换
   成 pod 真实 npz 路径（sha256 入册）再翻 true。
3. **groundfoot/push/force_push/qbar 四合同已确认**（wiring 全在钉住 commit
   `ad0110e8` 上——groundfoot 合并 `b9c8fff2` 与 push 波合并 `4624c824` 均已验证为其
   祖先，host 已 grep 白名单）；发射前仍须 grep 远端 checkout 复核（核对单第 2-4 条）。
4. **发射排位**：本波在全局队列排在 **franco_pipeline_20260722 战役完成且卡上有空槽
   之后**；空槽按 launch_order（combo_franco → combo_resume → combo_fresh，闸门未开
   跳过不阻塞）填。先 probe（2 updates smoke），一格通过即按序发全矩阵（精简治理）；
   同 pod boot 串行、错峰 ≥60 s。
5. **比值守卫核对**：probe + 首个里程碑按上表实测回填，超限停发（核对单第 14 条）。

## 谱系与合同声明

- 全波 `evidence_class=diagnostic_only_intentional_parent_contract_mismatch`：W 后代
  hard contract 有意 mismatch；grip/rough 键让新 checkpoint 长出 `ground_plant`
  schema-3 合同块（指纹在工作，不是事故）。
- **combo_fresh fresh 铁律**：平地 checkpoint 不得静默上粗糙地——渲染器断言 fresh 命令
  无任何 checkpoint 键；`ground_plant` 合同块 + resume `_contract_diff` 双保险。
- **集成波不做单项归因**：任何臂的胜出/失败都只回答"合体行不行"；单项证据永远以
  CGF/push/intel 单变量波为准。若 combo_resume 击球塌掉，处置是回到单变量读数定位
  元凶后再设计第二版合体，不是在本波上做减法穷举。
- **连续能力次序不被绕过**：本波是站立单拍配方合体，不改变连续能力的固定次序
  `T0`（按周期换题）→ `T1`（事件驱动结构、reward 冻结）→ `T2`（learned shaping）；
  任何臂的胜出都不授权跳过 T0/T1 直接上 T2 或连续演练。
- 本队列只做仿真（simulation_only），永不驱动真实机器人。
